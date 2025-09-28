package models

import (
    "fmt"
)

type ModelConfig struct {
    Algorithm    string
    K            int 
    Distance     string
    MaxDepth     int
    MinSplit     int
    NTrees       int
    VarSmoothing float64
}

func CreateModel(config ModelConfig) (Model, error) {
    switch config.Algorithm {
    case "knn":
        if config.K <= 0 {
            config.K = 5
        }
        if config.Distance == "" {
            config.Distance = "euclidean"
        }
        return NewKNN(config.K, config.Distance), nil

    case "tree":
        if config.MaxDepth <= 0 {
            config.MaxDepth = 10
        }
        if config.MinSplit <= 0 {
            config.MinSplit = 2
        }
        return NewDecisionTree(config.MaxDepth, config.MinSplit), nil

    case "forest":
        if config.NTrees <= 0 {
            config.NTrees = 100
        }
        if config.MaxDepth <= 0 {
            config.MaxDepth = 10
        }
        if config.MinSplit <= 0 {
            config.MinSplit = 2
        }
        return NewRandomForest(config.NTrees, config.MaxDepth, config.MinSplit), nil

    case "bayes":
        if config.VarSmoothing <= 0 {
            config.VarSmoothing = 1e-9
        }
        return NewNaiveBayes(config.VarSmoothing), nil

    default:
        return nil, fmt.Errorf("unknown algorithm: %s", config.Algorithm)
    }
}

func DefaultConfig(algorithm string) ModelConfig {
    config := ModelConfig{Algorithm: algorithm}

    switch algorithm {
    case "knn":
        config.K = 5
        config.Distance = "euclidean"
    case "tree":
        config.MaxDepth = 10
        config.MinSplit = 2
    case "forest":
        config.NTrees = 100
        config.MaxDepth = 10
        config.MinSplit = 2
    case "bayes":
        config.VarSmoothing = 1e-9
    }

    return config
}