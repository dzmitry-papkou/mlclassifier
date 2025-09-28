package models

import (
    "fmt"
    "math"
    "math/rand"
    "sync"

    "github.com/shopspring/decimal"
)

type RandomForest struct {
    BaseModel
    NTrees          int
    MaxDepth        int
    MinSamplesSplit int
    MaxFeatures     int
    Trees           []*DecisionTree
    FeatureIndices  [][]int
    Parallel        bool
    MaxWorkers      int
}

func NewRandomForest(nTrees, maxDepth, minSamplesSplit int) *RandomForest {
    return &RandomForest{
        NTrees:          nTrees,
        MaxDepth:        maxDepth,
        MinSamplesSplit: minSamplesSplit,
        Parallel:        true,
        MaxWorkers:      4,
        BaseModel: BaseModel{
            Name: "RandomForest",
            Params: map[string]any{
                "n_trees":          nTrees,
                "max_depth":        maxDepth,
                "min_samples_split": minSamplesSplit,
            },
        },
    }
}

func (rf *RandomForest) Fit(X [][]decimal.Decimal, y []int) error {
    rf.Classes = ExtractClasses(y)
    nFeatures := len(X[0])
    
    rf.MaxFeatures = int(math.Sqrt(float64(nFeatures)))
    if rf.MaxFeatures < 1 {
        rf.MaxFeatures = 1
    }
    
    rf.Trees = make([]*DecisionTree, rf.NTrees)
    rf.FeatureIndices = make([][]int, rf.NTrees)
    
    if rf.Parallel {
        return rf.trainParallel(X, y)
    }
    
    return rf.trainSequential(X, y)
}

func (rf *RandomForest) trainParallel(X [][]decimal.Decimal, y []int) error {
    var wg sync.WaitGroup
    errors := make([]error, rf.NTrees)
    
    workers := rf.MaxWorkers
    if workers > rf.NTrees {
        workers = rf.NTrees
    }
    
    jobs := make(chan int, rf.NTrees)
    
    for w := 0; w < workers; w++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for i := range jobs {
                tree, features, err := rf.trainSingleTree(X, y, int64(i))
                rf.Trees[i] = tree
                rf.FeatureIndices[i] = features
                errors[i] = err
            }
        }()
    }
    
    for i := 0; i < rf.NTrees; i++ {
        jobs <- i
    }
    close(jobs)
    
    wg.Wait()
    
    for i, err := range errors {
        if err != nil {
            return fmt.Errorf("tree %d training failed: %w", i, err)
        }
    }
    
    return nil
}

func (rf *RandomForest) trainSequential(X [][]decimal.Decimal, y []int) error {
    for i := 0; i < rf.NTrees; i++ {
        tree, features, err := rf.trainSingleTree(X, y, int64(i))
        if err != nil {
            return err
        }
        rf.Trees[i] = tree
        rf.FeatureIndices[i] = features
    }
    return nil
}

func (rf *RandomForest) trainSingleTree(X [][]decimal.Decimal, y []int, seed int64) (*DecisionTree, []int, error) {
    r := rand.New(rand.NewSource(seed))
    
    n := len(X)
    XBoot := make([][]decimal.Decimal, n)
    yBoot := make([]int, n)
    
    for i := 0; i < n; i++ {
        idx := r.Intn(n)
        XBoot[i] = X[idx]
        yBoot[i] = y[idx]
    }
    
    nFeatures := len(X[0])
    features := rf.selectRandomFeaturesOptimized(nFeatures, r)
    
    XSelected := make([][]decimal.Decimal, n)
    for i := range XBoot {
        XSelected[i] = make([]decimal.Decimal, len(features))
        for j, feat := range features {
            XSelected[i][j] = XBoot[i][feat]
        }
    }
    
    tree := NewDecisionTree(rf.MaxDepth, rf.MinSamplesSplit)
    err := tree.Fit(XSelected, yBoot)
    
    return tree, features, err
}

func (rf *RandomForest) selectRandomFeaturesOptimized(nFeatures int, r *rand.Rand) []int {
    features := make([]int, nFeatures)
    for i := range features {
        features[i] = i
    }
    
    for i := 0; i < rf.MaxFeatures && i < nFeatures; i++ {
        j := i + r.Intn(nFeatures-i)
        features[i], features[j] = features[j], features[i]
    }
    
    return features[:rf.MaxFeatures]
}

func (rf *RandomForest) Predict(X [][]decimal.Decimal) []int {
    predictions := make([]int, len(X))

    for i, sample := range X {
        votes := make(map[int]int)

        for j, tree := range rf.Trees {
            selectedSample := make([]decimal.Decimal, len(rf.FeatureIndices[j]))
            for k, feat := range rf.FeatureIndices[j] {
                selectedSample[k] = sample[feat]
            }

            treeInput := [][]decimal.Decimal{selectedSample}
            treePrediction := tree.Predict(treeInput)[0]
            votes[treePrediction]++
        }

        maxVotes := 0
        bestClass := rf.Classes[0]
        for class, count := range votes {
            if count > maxVotes {
                maxVotes = count
                bestClass = class
            }
        }

        predictions[i] = bestClass
    }

    return predictions
}

func (rf *RandomForest) PredictProba(X [][]decimal.Decimal) [][]decimal.Decimal {
    proba := make([][]decimal.Decimal, len(X))

    for i, sample := range X {
        classProba := make(map[int]decimal.Decimal)
        for _, class := range rf.Classes {
            classProba[class] = decimal.Zero
        }

        for j, tree := range rf.Trees {
            selectedSample := make([]decimal.Decimal, len(rf.FeatureIndices[j]))
            for k, feat := range rf.FeatureIndices[j] {
                selectedSample[k] = sample[feat]
            }

            treeInput := [][]decimal.Decimal{selectedSample}
            treePrediction := tree.Predict(treeInput)[0]
            classProba[treePrediction] = classProba[treePrediction].Add(decimal.NewFromInt(1))
        }

        proba[i] = make([]decimal.Decimal, len(rf.Classes))
        nTrees := decimal.NewFromInt(int64(rf.NTrees))
        for j, class := range rf.Classes {
            proba[i][j] = classProba[class].Div(nTrees)
        }
    }

    return proba
}

func (rf *RandomForest) GetClasses() []int {
    return rf.Classes
}

func (rf *RandomForest) Reset() {
    rf.Trees = nil
    rf.FeatureIndices = nil
    rf.Classes = nil
}