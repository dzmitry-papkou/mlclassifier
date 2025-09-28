package evaluation

import (
    "fmt"
    "math"
    "sync"
    
    "mlclassifier/internal/models"
    "github.com/shopspring/decimal"
)

type CrossValidator struct {
    NFolds       int
    Stratified   bool
    Shuffle      bool
    RandomSeed   int64
    Parallel     bool
    MaxWorkers   int
}

func NewCrossValidator(nFolds int, stratified bool) *CrossValidator {
    return &CrossValidator{
        NFolds:     nFolds,
        Stratified: stratified,
        Shuffle:    true,
        RandomSeed: 42,
        Parallel:   true,
        MaxWorkers: 4,
    }
}

func (cv *CrossValidator) ParallelCrossValidate(
    X [][]decimal.Decimal,
    y []int,
    model models.Model,
) ([]float64, float64, float64, error) {
    
    if !cv.Parallel {
        return cv.CrossValidateSerial(X, y, model)
    }
    
    folds, err := cv.KFoldSplit(len(X), y)
    if err != nil {
        return nil, 0, 0, err
    }
    
    scores := make([]float64, cv.NFolds)
    errors := make([]error, cv.NFolds)
    
    workers := cv.MaxWorkers
    if workers > cv.NFolds {
        workers = cv.NFolds
    }
    
    type foldJob struct {
        index int
        testIndices []int
    }
    
    jobs := make(chan foldJob, cv.NFolds)
    var wg sync.WaitGroup
    
    for w := 0; w < workers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                score, err := cv.evaluateFold(X, y, model, job.testIndices)
                scores[job.index] = score
                errors[job.index] = err
            }
        }()
    }
    
    for i, fold := range folds {
        jobs <- foldJob{index: i, testIndices: fold}
    }
    close(jobs)
    
    wg.Wait()
    
    for i, err := range errors {
        if err != nil {
            return nil, 0, 0, fmt.Errorf("fold %d failed: %w", i, err)
        }
    }
    
    mean, std := cv.calculateStats(scores)
    return scores, mean, std, nil
}

func (cv *CrossValidator) evaluateFold(
    X [][]decimal.Decimal,
    y []int,
    model models.Model,
    testIndices []int,
) (float64, error) {
    
    testSet := make(map[int]bool)
    for _, idx := range testIndices {
        testSet[idx] = true
    }
    
    trainIndices := make([]int, 0, len(X)-len(testIndices))
    for i := 0; i < len(X); i++ {
        if !testSet[i] {
            trainIndices = append(trainIndices, i)
        }
    }
    
    XTrain := make([][]decimal.Decimal, len(trainIndices))
    yTrain := make([]int, len(trainIndices))
    for i, idx := range trainIndices {
        XTrain[i] = X[idx]
        yTrain[i] = y[idx]
    }
    
    XTest := make([][]decimal.Decimal, len(testIndices))
    yTest := make([]int, len(testIndices))
    for i, idx := range testIndices {
        XTest[i] = X[idx]
        yTest[i] = y[idx]
    }
    
    foldModel := cv.cloneModel(model)
    if err := foldModel.Fit(XTrain, yTrain); err != nil {
        return 0, err
    }
    
    predictions := foldModel.Predict(XTest)
    
    correct := 0
    for i, pred := range predictions {
        if pred == yTest[i] {
            correct++
        }
    }
    
    return float64(correct) / float64(len(yTest)), nil
}

func (cv *CrossValidator) calculateStats(scores []float64) (mean, std float64) {
    if len(scores) == 0 {
        return 0, 0
    }
    
    sum := 0.0
    for _, s := range scores {
        sum += s
    }
    mean = sum / float64(len(scores))
    
    if len(scores) > 1 {
        variance := 0.0
        for _, s := range scores {
            diff := s - mean
            variance += diff * diff
        }
        variance /= float64(len(scores) - 1)
        std = math.Sqrt(variance)
    }
    
    return mean, std
}

func (cv *CrossValidator) CrossValidateSerial(
    X [][]decimal.Decimal,
    y []int,
    model models.Model,
) ([]float64, float64, float64, error) {

    folds, err := cv.KFoldSplit(len(X), y)
    if err != nil {
        return nil, 0, 0, err
    }

    scores := make([]float64, cv.NFolds)

    for i, testIndices := range folds {
        score, err := cv.evaluateFold(X, y, model, testIndices)
        if err != nil {
            return nil, 0, 0, fmt.Errorf("fold %d failed: %v", i, err)
        }
        scores[i] = score
    }

    mean, std := cv.calculateStats(scores)
    return scores, mean, std, nil
}

func (cv *CrossValidator) KFoldSplit(n int, y []int) ([][]int, error) {
    if cv.NFolds < 2 || cv.NFolds > n {
        return nil, fmt.Errorf("invalid number of folds: %d (must be between 2 and %d)", cv.NFolds, n)
    }

    indices := make([]int, n)
    for i := range indices {
        indices[i] = i
    }

    if cv.Shuffle {
        cv.shuffleIndices(indices)
    }

    folds := make([][]int, cv.NFolds)
    foldSize := n / cv.NFolds

    for i := 0; i < cv.NFolds; i++ {
        start := i * foldSize
        end := start + foldSize
        if i == cv.NFolds-1 {
            end = n
        }

        folds[i] = make([]int, end-start)
        copy(folds[i], indices[start:end])
    }

    return folds, nil
}

func (cv *CrossValidator) cloneModel(model models.Model) models.Model {
    modelType := model.GetType()
    params := model.GetParams()

    switch modelType {
    case "KNN":
        k := params["k"].(int)
        distance := params["distance"].(string)
        return models.NewKNN(k, distance)
    case "DecisionTree":
        maxDepth := params["max_depth"].(int)
        minSamplesSplit := params["min_samples_split"].(int)
        return models.NewDecisionTree(maxDepth, minSamplesSplit)
    case "RandomForest":
        nTrees := params["n_trees"].(int)
        maxDepth := params["max_depth"].(int)
        minSamplesSplit := params["min_samples_split"].(int)
        return models.NewRandomForest(nTrees, maxDepth, minSamplesSplit)
    case "NaiveBayes":
        varSmoothing := params["var_smoothing"].(float64)
        return models.NewNaiveBayes(varSmoothing)
    default:
        model.Reset()
        return model
    }
}

func (cv *CrossValidator) shuffleIndices(indices []int) {
    for i := len(indices) - 1; i > 0; i-- {
        seed := cv.RandomSeed
        if seed < 0 {
            seed = -seed
        }
        j := int(seed) % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
        cv.RandomSeed = cv.RandomSeed*1103515245 + 12345
    }
}