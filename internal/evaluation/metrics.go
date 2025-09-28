package evaluation

import (
    "fmt"
    "math"
)

type ClassificationMetrics struct {
    Accuracy         float64            `json:"accuracy"`
    BalancedAccuracy float64            `json:"balanced_accuracy"`
    WeightedAccuracy float64            `json:"weighted_accuracy"`
    Precision        float64            `json:"precision"`
    Recall           float64            `json:"recall"`
    F1Score          float64            `json:"f1_score"`
    MacroPrecision   float64            `json:"macro_precision"`
    MacroRecall      float64            `json:"macro_recall"`
    MacroF1          float64            `json:"macro_f1"`
    MicroPrecision   float64            `json:"micro_precision"`
    MicroRecall      float64            `json:"micro_recall"`
    MicroF1          float64            `json:"micro_f1"`
    WeightedPrecision float64           `json:"weighted_precision"`
    WeightedRecall   float64            `json:"weighted_recall"`
    WeightedF1       float64            `json:"weighted_f1"`
    PerClassMetrics  map[int]ClassMetrics `json:"per_class_metrics"`
    ConfusionMatrix  [][]int            `json:"confusion_matrix"`
    ClassSupport     map[int]int        `json:"class_support"`
    NumSamples       int                `json:"num_samples"`
    NumClasses       int                `json:"num_classes"`
}

type ClassMetrics struct {
    Precision   float64 `json:"precision"`
    Recall      float64 `json:"recall"`
    F1Score     float64 `json:"f1_score"`
    Specificity float64 `json:"specificity"`
    Support     int     `json:"support"`
}

func CalculateMetrics(yTrue, yPred []int, classes []int) *ClassificationMetrics {
    if len(yTrue) != len(yPred) {
        return nil
    }

    numSamples := len(yTrue)
    numClasses := len(classes)

    confusionMatrix := buildConfusionMatrix(yTrue, yPred, classes)

    classSupport := make(map[int]int)
    for _, class := range yTrue {
        classSupport[class]++
    }

    perClassMetrics := make(map[int]ClassMetrics)
    var macroPrec, macroRec, macroF1, macroSpec float64
    var weightedPrec, weightedRec, weightedF1 float64
    totalSupport := 0

    for i, class := range classes {
        tp := confusionMatrix[i][i]
        fp := 0
        fn := 0
        tn := 0

        for j := range classes {
            if j != i {
                fp += confusionMatrix[j][i]
                fn += confusionMatrix[i][j]
            }
        }

        for j := range classes {
            for k := range classes {
                if j != i && k != i {
                    tn += confusionMatrix[j][k]
                }
            }
        }

        precision := safeDivide(float64(tp), float64(tp+fp))
        recall := safeDivide(float64(tp), float64(tp+fn))
        f1 := safeDivide(2*precision*recall, precision+recall)
        specificity := safeDivide(float64(tn), float64(tn+fp))

        support := classSupport[class]
        perClassMetrics[class] = ClassMetrics{
            Precision:   precision,
            Recall:      recall,
            F1Score:     f1,
            Specificity: specificity,
            Support:     support,
        }

        macroPrec += precision
        macroRec += recall
        macroF1 += f1
        macroSpec += specificity

        weightedPrec += precision * float64(support)
        weightedRec += recall * float64(support)
        weightedF1 += f1 * float64(support)
        totalSupport += support
    }

    macroPrec /= float64(numClasses)
    macroRec /= float64(numClasses)
    macroF1 /= float64(numClasses)

    weightedPrec /= float64(totalSupport)
    weightedRec /= float64(totalSupport)
    weightedF1 /= float64(totalSupport)

    correct := 0
    for i, pred := range yPred {
        if pred == yTrue[i] {
            correct++
        }
    }
    accuracy := float64(correct) / float64(numSamples)

    balancedAccuracy := 0.0
    for _, class := range classes {
        balancedAccuracy += perClassMetrics[class].Recall
    }
    balancedAccuracy /= float64(numClasses)

    return &ClassificationMetrics{
        Accuracy:         accuracy,
        BalancedAccuracy: balancedAccuracy,
        WeightedAccuracy: weightedRec,
        Precision:        macroPrec,
        Recall:           macroRec,
        F1Score:          macroF1,
        MacroPrecision:   macroPrec,
        MacroRecall:      macroRec,
        MacroF1:          macroF1,
        MicroPrecision:   accuracy,
        MicroRecall:      accuracy,
        MicroF1:          accuracy,
        WeightedPrecision: weightedPrec,
        WeightedRecall:   weightedRec,
        WeightedF1:       weightedF1,
        PerClassMetrics:  perClassMetrics,
        ConfusionMatrix:  confusionMatrix,
        ClassSupport:     classSupport,
        NumSamples:       numSamples,
        NumClasses:       numClasses,
    }
}

func buildConfusionMatrix(yTrue, yPred []int, classes []int) [][]int {
    numClasses := len(classes)
    matrix := make([][]int, numClasses)
    for i := range matrix {
        matrix[i] = make([]int, numClasses)
    }

    classToIdx := make(map[int]int)
    for i, class := range classes {
        classToIdx[class] = i
    }

    for i := range yTrue {
        trueIdx, trueOk := classToIdx[yTrue[i]]
        predIdx, predOk := classToIdx[yPred[i]]
        if trueOk && predOk {
            matrix[trueIdx][predIdx]++
        }
    }

    return matrix
}

func safeDivide(numerator, denominator float64) float64 {
    if denominator == 0 {
        return 0.0
    }
    result := numerator / denominator
    if math.IsNaN(result) || math.IsInf(result, 0) {
        return 0.0
    }
    return result
}

func (m *ClassificationMetrics) FormatMetrics() string {
    result := fmt.Sprintf("Accuracy: %.4f\n", m.Accuracy)
    result += fmt.Sprintf("Balanced Accuracy: %.4f\n", m.BalancedAccuracy)
    result += fmt.Sprintf("Macro Avg - Precision: %.4f, Recall: %.4f, F1: %.4f\n",
        m.MacroPrecision, m.MacroRecall, m.MacroF1)
    result += fmt.Sprintf("Weighted Avg - Precision: %.4f, Recall: %.4f, F1: %.4f\n",
        m.WeightedPrecision, m.WeightedRecall, m.WeightedF1)
    return result
}