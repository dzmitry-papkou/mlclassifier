package models

import (
    "math"
    "sort"

    "github.com/shopspring/decimal"
)

type KNN struct {
    BaseModel
    K           int
    Distance    string
    XTrain      [][]decimal.Decimal
    yTrain      []int
}

func NewKNN(k int, distance string) *KNN {
    if k <= 0 {
        k = 5
    }

    if distance != "euclidean" && distance != "manhattan" {
        distance = "euclidean"
    }

    return &KNN{
        K:        k,
        Distance: distance,
        BaseModel: BaseModel{
            Name: "KNN",
            Params: map[string]any{
                "k":        k,
                "distance": distance,
            },
        },
    }
}

func (knn *KNN) Fit(X [][]decimal.Decimal, y []int) error {
    knn.XTrain = make([][]decimal.Decimal, len(X))
    for i := range X {
        knn.XTrain[i] = make([]decimal.Decimal, len(X[i]))
        copy(knn.XTrain[i], X[i])
    }

    knn.yTrain = make([]int, len(y))
    copy(knn.yTrain, y)

    knn.Classes = ExtractClasses(y)
    return nil
}

func (knn *KNN) Predict(X [][]decimal.Decimal) []int {
    predictions := make([]int, len(X))

    for i, sample := range X {
        neighbors := knn.findNeighbors(sample)
        predictions[i] = knn.majorityVote(neighbors)
    }

    return predictions
}

func (knn *KNN) PredictProba(X [][]decimal.Decimal) [][]decimal.Decimal {
    proba := make([][]decimal.Decimal, len(X))

    for i, sample := range X {
        neighbors := knn.findNeighbors(sample)
        proba[i] = knn.calculateProbabilities(neighbors)
    }

    return proba
}

func (knn *KNN) findNeighbors(sample []decimal.Decimal) []int {
    type neighbor struct {
        index    int
        distance float64
    }

    neighbors := make([]neighbor, len(knn.XTrain))

    for i, trainSample := range knn.XTrain {
        dist := knn.calculateDistance(sample, trainSample)
        neighbors[i] = neighbor{index: i, distance: dist}
    }

    sort.Slice(neighbors, func(i, j int) bool {
        return neighbors[i].distance < neighbors[j].distance
    })

    kNeighbors := make([]int, knn.K)
    for i := 0; i < knn.K && i < len(neighbors); i++ {
        kNeighbors[i] = neighbors[i].index
    }

    return kNeighbors
}

func (knn *KNN) calculateDistance(a, b []decimal.Decimal) float64 {
    switch knn.Distance {
    case "euclidean":
        sum := 0.0
        for i := range a {
            diff, _ := a[i].Sub(b[i]).Float64()
            sum += diff * diff
        }
        return math.Sqrt(sum)
    case "manhattan":
        sum := 0.0
        for i := range a {
            diff, _ := a[i].Sub(b[i]).Abs().Float64()
            sum += diff
        }
        return sum
    default:
        return knn.calculateDistance(a, b)
    }
}

func (knn *KNN) majorityVote(neighbors []int) int {
    votes := make(map[int]int)

    for _, neighborIdx := range neighbors {
        if neighborIdx < len(knn.yTrain) {
            class := knn.yTrain[neighborIdx]
            votes[class]++
        }
    }

    maxVotes := 0
    bestClass := knn.Classes[0]

    for class, count := range votes {
        if count > maxVotes {
            maxVotes = count
            bestClass = class
        }
    }

    return bestClass
}

func (knn *KNN) calculateProbabilities(neighbors []int) []decimal.Decimal {
    votes := make(map[int]int)

    for _, neighborIdx := range neighbors {
        if neighborIdx < len(knn.yTrain) {
            class := knn.yTrain[neighborIdx]
            votes[class]++
        }
    }

    proba := make([]decimal.Decimal, len(knn.Classes))
    totalVotes := decimal.NewFromInt(int64(len(neighbors)))

    for i, class := range knn.Classes {
        count := votes[class]
        proba[i] = decimal.NewFromInt(int64(count)).Div(totalVotes)
    }

    return proba
}

func (knn *KNN) GetClasses() []int {
    return knn.Classes
}

func (knn *KNN) Reset() {
    knn.XTrain = nil
    knn.yTrain = nil
    knn.Classes = nil
}