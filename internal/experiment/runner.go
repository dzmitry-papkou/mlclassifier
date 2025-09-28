package experiment

import (
    "encoding/csv"
    "fmt"
    "os"
    "time"
    
    "mlclassifier/internal/evaluation"
    "mlclassifier/internal/models"
    "mlclassifier/internal/preprocessing"
    
    "github.com/shopspring/decimal"
    "gopkg.in/yaml.v3"
)

type ExperimentRunner struct {
    Config *ExperimentConfig
}

type ExperimentConfig struct {
    Experiment struct {
        Preprocessing   []string  `yaml:"preprocessing"`
        TrainTestSplits []float64 `yaml:"train_test_splits"`
        CrossValidation struct {
            Folds int `yaml:"folds"`
        } `yaml:"cross_validation"`
        Algorithms struct {
            KNN struct {
                K        []int    `yaml:"k"`
                Distance []string `yaml:"distance"`
            } `yaml:"knn"`
            DecisionTree struct {
                MaxDepth         []int `yaml:"max_depth"`
                MinSamplesSplit  []int `yaml:"min_samples_split"`
            } `yaml:"decision_tree"`
            RandomForest struct {
                NTrees   []int `yaml:"n_trees"`
                MaxDepth []int `yaml:"max_depth"`
            } `yaml:"random_forest"`
            NaiveBayes struct {
                VarSmoothing []float64 `yaml:"var_smoothing"`
            } `yaml:"naive_bayes"`
        } `yaml:"algorithms"`
    } `yaml:"experiment"`
}

func NewRunner(configFile string) *ExperimentRunner {
    config := &ExperimentConfig{}
    
    data, err := os.ReadFile(configFile)
    if err == nil {
        yaml.Unmarshal(data, config)
    }
    
    return &ExperimentRunner{Config: config}
}

func (r *ExperimentRunner) RunAllExperiments(dataFile string) ([]ExperimentResult, error) {
    X, y, _, err := r.loadData(dataFile)
    if err != nil {
        return nil, err
    }
    
    var results []ExperimentResult
    
    for _, prepMethod := range r.Config.Experiment.Preprocessing {
        XProcessed := r.preprocess(X, prepMethod)
        
        for _, split := range r.Config.Experiment.TrainTestSplits {
            results = append(results, r.testKNN(XProcessed, y, prepMethod, split)...)
            results = append(results, r.testDecisionTree(XProcessed, y, prepMethod, split)...)
            results = append(results, r.testRandomForest(XProcessed, y, prepMethod, split)...)
            results = append(results, r.testNaiveBayes(XProcessed, y, prepMethod, split)...)
        }
    }
    
    return results, nil
}

func (r *ExperimentRunner) loadData(filename string) ([][]decimal.Decimal, []int, []string, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, nil, nil, err
    }
    defer file.Close()
    
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        return nil, nil, nil, err
    }
    
    headers := records[0]
    data := records[1:]
    
    X := make([][]decimal.Decimal, len(data))
    labels := make([]string, len(data))
    
    for i, record := range data {
        X[i] = make([]decimal.Decimal, len(record)-1)
        for j := 0; j < len(record)-1; j++ {
            val, _ := decimal.NewFromString(record[j])
            X[i][j] = val
        }
        labels[i] = record[len(record)-1]
    }
    
    encoder := preprocessing.NewLabelEncoder()
    y, _ := encoder.FitTransform(labels)
    
    return X, y, headers[:len(headers)-1], nil
}

func (r *ExperimentRunner) preprocess(X [][]decimal.Decimal, method string) [][]decimal.Decimal {
    switch method {
    case "normalized":
        scaler := preprocessing.NewScaler("minmax")
        result, err := scaler.FitTransform(X)
        if err != nil {
            return X
        }
        return result
    case "standardized":
        scaler := preprocessing.NewScaler("standard")
        result, err := scaler.FitTransform(X)
        if err != nil {
            return X
        }
        return result
    default:
        return X
    }
}

func (r *ExperimentRunner) testKNN(X [][]decimal.Decimal, y []int, prep string, split float64) []ExperimentResult {
    var results []ExperimentResult
    
    for _, k := range r.Config.Experiment.Algorithms.KNN.K {
        for _, dist := range r.Config.Experiment.Algorithms.KNN.Distance {
            model := models.NewKNN(k, dist)
            result := r.evaluateModel(model, X, y, prep, split)
            result.Algorithm = "KNN"
            results = append(results, result)
        }
    }
    
    return results
}

func (r *ExperimentRunner) testDecisionTree(X [][]decimal.Decimal, y []int, prep string, split float64) []ExperimentResult {
    var results []ExperimentResult
    
    for _, depth := range r.Config.Experiment.Algorithms.DecisionTree.MaxDepth {
        for _, minSplit := range r.Config.Experiment.Algorithms.DecisionTree.MinSamplesSplit {
            model := models.NewDecisionTree(depth, minSplit)
            result := r.evaluateModel(model, X, y, prep, split)
            result.Algorithm = "DecisionTree"
            results = append(results, result)
        }
    }
    
    return results
}

func (r *ExperimentRunner) testRandomForest(X [][]decimal.Decimal, y []int, prep string, split float64) []ExperimentResult {
    var results []ExperimentResult
    
    for _, nTrees := range r.Config.Experiment.Algorithms.RandomForest.NTrees {
        for _, depth := range r.Config.Experiment.Algorithms.RandomForest.MaxDepth {
            model := models.NewRandomForest(nTrees, depth, 2)
            result := r.evaluateModel(model, X, y, prep, split)
            result.Algorithm = "RandomForest"
            results = append(results, result)
        }
    }
    
    return results
}

func (r *ExperimentRunner) testNaiveBayes(X [][]decimal.Decimal, y []int, prep string, split float64) []ExperimentResult {
    var results []ExperimentResult
    
    for _, smooth := range r.Config.Experiment.Algorithms.NaiveBayes.VarSmoothing {
        model := models.NewNaiveBayes(smooth)
        result := r.evaluateModel(model, X, y, prep, split)
        result.Algorithm = "NaiveBayes"
        results = append(results, result)
    }
    
    return results
}

func (r *ExperimentRunner) evaluateModel(model models.Model, X [][]decimal.Decimal, y []int, prep string, split float64) ExperimentResult {
    result := ExperimentResult{
        Parameters:    fmt.Sprintf("%v", model.GetParams()),
        Preprocessing: prep,
        TrainTestSplit: fmt.Sprintf("%.0f-%.0f", split*100, (1-split)*100),
    }
    
    splitIdx := int(float64(len(X)) * split)
    XTrain := X[:splitIdx]
    XTest := X[splitIdx:]
    yTrain := y[:splitIdx]
    yTest := y[splitIdx:]
    
    startTime := time.Now()
    model.Fit(XTrain, yTrain)
    result.TrainingTimeMs = time.Since(startTime).Milliseconds()
    
    predictions := model.Predict(XTest)
    classes := models.ExtractClasses(y)
    metrics := evaluation.CalculateMetrics(yTest, predictions, classes)
    
    result.Accuracy = metrics.Accuracy
    result.Precision = metrics.MacroPrecision
    result.Recall = metrics.MacroRecall
    result.F1Score = metrics.MacroF1
    
    if r.Config.Experiment.CrossValidation.Folds > 0 {
        cv := evaluation.NewCrossValidator(r.Config.Experiment.CrossValidation.Folds, true)
        _, mean, std, err := cv.ParallelCrossValidate(X, y, model)
        if err != nil {
            _, mean, std, _ = cv.CrossValidateSerial(X, y, model)
        }
        result.CVMean = mean
        result.CVStd = std
    }
    
    return result
}

type ExperimentResult struct {
    Dataset        string
    Algorithm      string
    Parameters     string
    Preprocessing  string
    TrainTestSplit string
    Accuracy       float64
    Precision      float64
    Recall         float64
    F1Score        float64
    CVMean         float64
    CVStd          float64
    TrainingTimeMs int64
}

func (r *ExperimentRunner) ExportResults(results []ExperimentResult, filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    writer := csv.NewWriter(file)
    defer writer.Flush()
    
    writer.Write([]string{
        "Dataset", "Algorithm", "Parameters", "Preprocessing",
        "TrainTestSplit", "Accuracy", "Precision", "Recall", "F1Score",
        "CVMean", "CVStd", "TrainingTimeMs",
    })
    
    for _, result := range results {
        writer.Write([]string{
            result.Dataset,
            result.Algorithm,
            result.Parameters,
            result.Preprocessing,
            result.TrainTestSplit,
            fmt.Sprintf("%.4f", result.Accuracy),
            fmt.Sprintf("%.4f", result.Precision),
            fmt.Sprintf("%.4f", result.Recall),
            fmt.Sprintf("%.4f", result.F1Score),
            fmt.Sprintf("%.4f", result.CVMean),
            fmt.Sprintf("%.4f", result.CVStd),
            fmt.Sprintf("%d", result.TrainingTimeMs),
        })
    }
    
    return nil
}