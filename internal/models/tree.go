package models

import (
    "github.com/shopspring/decimal"
)

type TreeNode struct {
    IsLeaf          bool
    Class           int
    Feature         int
    Threshold       decimal.Decimal
    Left            *TreeNode
    Right           *TreeNode
    Samples         int
    Impurity        float64
    ImpurityDecrease float64
}

type DecisionTree struct {
    BaseModel
    Root               *TreeNode
    MaxDepth           int
    MinSamplesSplit    int
    MinImpurityDecrease float64
    EnablePruning      bool
}

func NewDecisionTree(maxDepth, minSamplesSplit int) *DecisionTree {
    if maxDepth <= 0 {
        maxDepth = 10
    }

    if minSamplesSplit <= 0 {
        minSamplesSplit = 2
    }

    return &DecisionTree{
        MaxDepth:           maxDepth,
        MinSamplesSplit:    minSamplesSplit,
        MinImpurityDecrease: 0.01,
        EnablePruning:      true,
        BaseModel: BaseModel{
            Name: "DecisionTree",
            Params: map[string]any{
                "max_depth":        maxDepth,
                "min_samples_split": minSamplesSplit,
            },
        },
    }
}

func (dt *DecisionTree) buildTree(X [][]decimal.Decimal, y []int, depth int) *TreeNode {
    node := &TreeNode{
        Samples: len(y),
    }
    
    node.Impurity = dt.calculateGini(y)
    
    if depth >= dt.MaxDepth || 
       len(y) < dt.MinSamplesSplit || 
       dt.isPure(y) ||
       node.Impurity < dt.MinImpurityDecrease {
        
        node.IsLeaf = true
        node.Class = dt.mostCommonClass(y)
        return node
    }
    
    bestFeature, bestThreshold, bestImpurityDecrease := dt.findBestSplitWithPruning(X, y)
    
    if bestImpurityDecrease < dt.MinImpurityDecrease {
        node.IsLeaf = true
        node.Class = dt.mostCommonClass(y)
        return node
    }
    
    node.Feature = bestFeature
    node.Threshold = bestThreshold
    node.ImpurityDecrease = bestImpurityDecrease
    
    leftIndices, rightIndices := dt.splitData(X, bestFeature, bestThreshold)
    
    if len(leftIndices) == 0 || len(rightIndices) == 0 {
        node.IsLeaf = true
        node.Class = dt.mostCommonClass(y)
        return node
    }
    
    XLeft, yLeft := dt.selectData(X, y, leftIndices)
    XRight, yRight := dt.selectData(X, y, rightIndices)
    
    node.Left = dt.buildTree(XLeft, yLeft, depth+1)
    node.Right = dt.buildTree(XRight, yRight, depth+1)
    
    return node
}

func (dt *DecisionTree) findBestSplitWithPruning(X [][]decimal.Decimal, y []int) (int, decimal.Decimal, float64) {
    bestFeature := 0
    bestThreshold := decimal.Zero
    bestImpurityDecrease := 0.0
    
    parentImpurity := dt.calculateGini(y)
    n := len(y)
    
    for feature := range X[0] {
        thresholds := dt.getUniqueValues(X, feature)
        
        for _, threshold := range thresholds {
            leftIndices, rightIndices := dt.splitData(X, feature, threshold)
            
            if len(leftIndices) == 0 || len(rightIndices) == 0 {
                continue
            }
            
            yLeft := make([]int, len(leftIndices))
            yRight := make([]int, len(rightIndices))
            
            for i, idx := range leftIndices {
                yLeft[i] = y[idx]
            }
            for i, idx := range rightIndices {
                yRight[i] = y[idx]
            }
            
            leftImpurity := dt.calculateGini(yLeft)
            rightImpurity := dt.calculateGini(yRight)
            
            weightedImpurity := (float64(len(leftIndices))/float64(n))*leftImpurity +
                               (float64(len(rightIndices))/float64(n))*rightImpurity
            
            impurityDecrease := parentImpurity - weightedImpurity
            
            if impurityDecrease > bestImpurityDecrease {
                bestImpurityDecrease = impurityDecrease
                bestFeature = feature
                bestThreshold = threshold
            }
        }
    }
    
    return bestFeature, bestThreshold, bestImpurityDecrease
}

func (dt *DecisionTree) Prune(XVal [][]decimal.Decimal, yVal []int) {
    if !dt.EnablePruning || dt.Root == nil {
        return
    }
    
    dt.pruneNode(dt.Root, XVal, yVal)
}

func (dt *DecisionTree) pruneNode(node *TreeNode, XVal [][]decimal.Decimal, yVal []int) float64 {
    if node.IsLeaf {
        return dt.calculateAccuracy(node, XVal, yVal)
    }
    
    accuracyWithSubtrees := dt.calculateAccuracy(node, XVal, yVal)
    
    originalLeft := node.Left
    originalRight := node.Right
    node.IsLeaf = true
    node.Class = dt.predictLeaf(node)
    
    accuracyAsLeaf := dt.calculateAccuracy(node, XVal, yVal)
    
    if accuracyAsLeaf >= accuracyWithSubtrees {
        node.Left = nil
        node.Right = nil
    } else {
        node.IsLeaf = false
        node.Left = originalLeft
        node.Right = originalRight
        
        if node.Left != nil {
            dt.pruneNode(node.Left, XVal, yVal)
        }
        if node.Right != nil {
            dt.pruneNode(node.Right, XVal, yVal)
        }
    }
    
    return accuracyAsLeaf
}

func (dt *DecisionTree) Fit(X [][]decimal.Decimal, y []int) error {
    dt.Classes = ExtractClasses(y)
    dt.Root = dt.buildTree(X, y, 0)
    return nil
}

func (dt *DecisionTree) Predict(X [][]decimal.Decimal) []int {
    predictions := make([]int, len(X))

    for i, sample := range X {
        predictions[i] = dt.predictSample(sample, dt.Root)
    }

    return predictions
}

func (dt *DecisionTree) PredictProba(X [][]decimal.Decimal) [][]decimal.Decimal {
    proba := make([][]decimal.Decimal, len(X))

    for i, sample := range X {
        prediction := dt.predictSample(sample, dt.Root)
        proba[i] = make([]decimal.Decimal, len(dt.Classes))

        for j, class := range dt.Classes {
            if class == prediction {
                proba[i][j] = decimal.NewFromInt(1)
            } else {
                proba[i][j] = decimal.Zero
            }
        }
    }

    return proba
}

func (dt *DecisionTree) predictSample(sample []decimal.Decimal, node *TreeNode) int {
    if node.IsLeaf {
        return node.Class
    }

    if sample[node.Feature].LessThan(node.Threshold) {
        return dt.predictSample(sample, node.Left)
    } else {
        return dt.predictSample(sample, node.Right)
    }
}

func (dt *DecisionTree) GetClasses() []int {
    return dt.Classes
}

func (dt *DecisionTree) Reset() {
    dt.Root = nil
    dt.Classes = nil
}

func (dt *DecisionTree) calculateGini(y []int) float64 {
    if len(y) == 0 {
        return 0.0
    }

    classCounts := make(map[int]int)
    for _, class := range y {
        classCounts[class]++
    }

    impurity := 1.0
    n := float64(len(y))

    for _, count := range classCounts {
        p := float64(count) / n
        impurity -= p * p
    }

    return impurity
}

func (dt *DecisionTree) isPure(y []int) bool {
    if len(y) == 0 {
        return true
    }

    firstClass := y[0]
    for _, class := range y {
        if class != firstClass {
            return false
        }
    }

    return true
}

func (dt *DecisionTree) mostCommonClass(y []int) int {
    if len(y) == 0 {
        return 0
    }

    classCounts := make(map[int]int)
    for _, class := range y {
        classCounts[class]++
    }

    maxCount := 0
    mostCommon := y[0]

    for class, count := range classCounts {
        if count > maxCount {
            maxCount = count
            mostCommon = class
        }
    }

    return mostCommon
}

func (dt *DecisionTree) getUniqueValues(X [][]decimal.Decimal, feature int) []decimal.Decimal {
    valueMap := make(map[string]decimal.Decimal)

    for _, sample := range X {
        key := sample[feature].String()
        valueMap[key] = sample[feature]
    }

    values := make([]decimal.Decimal, 0, len(valueMap))
    for _, value := range valueMap {
        values = append(values, value)
    }

    return values
}

func (dt *DecisionTree) splitData(X [][]decimal.Decimal, feature int, threshold decimal.Decimal) ([]int, []int) {
    var leftIndices, rightIndices []int

    for i, sample := range X {
        if sample[feature].LessThan(threshold) {
            leftIndices = append(leftIndices, i)
        } else {
            rightIndices = append(rightIndices, i)
        }
    }

    return leftIndices, rightIndices
}

func (dt *DecisionTree) selectData(X [][]decimal.Decimal, y []int, indices []int) ([][]decimal.Decimal, []int) {
    selectedX := make([][]decimal.Decimal, len(indices))
    selectedY := make([]int, len(indices))

    for i, idx := range indices {
        selectedX[i] = X[idx]
        selectedY[i] = y[idx]
    }

    return selectedX, selectedY
}

func (dt *DecisionTree) calculateAccuracy(node *TreeNode, XVal [][]decimal.Decimal, yVal []int) float64 {
    if len(XVal) == 0 {
        return 0.0
    }

    correct := 0
    for i, sample := range XVal {
        prediction := dt.predictSample(sample, node)
        if prediction == yVal[i] {
            correct++
        }
    }

    return float64(correct) / float64(len(XVal))
}

func (dt *DecisionTree) predictLeaf(node *TreeNode) int {
    return node.Class
}