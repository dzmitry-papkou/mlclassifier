package models

import (
    "fmt"
    "math"
    
    "github.com/shopspring/decimal"
)

type NaiveBayes struct {
    BaseModel
    ClassLogPriors map[int]float64
    FeatureMeans   map[int][]decimal.Decimal
    FeatureVars    map[int][]decimal.Decimal
    VarSmoothing   decimal.Decimal
    UseLogSpace    bool
}

func NewNaiveBayes(varSmoothing float64) *NaiveBayes {
    return &NaiveBayes{
        VarSmoothing: decimal.NewFromFloat(varSmoothing),
        UseLogSpace:  true,
        BaseModel: BaseModel{
            Name: "NaiveBayes",
            Params: map[string]any{
                "var_smoothing": varSmoothing,
            },
        },
    }
}

func (nb *NaiveBayes) Fit(X [][]decimal.Decimal, y []int) error {
    nb.Classes = ExtractClasses(y)
    nFeatures := len(X[0])
    
    nb.ClassLogPriors = make(map[int]float64)
    nb.FeatureMeans = make(map[int][]decimal.Decimal)
    nb.FeatureVars = make(map[int][]decimal.Decimal)
    
    for _, class := range nb.Classes {
        classData := [][]decimal.Decimal{}
        for i, label := range y {
            if label == class {
                classData = append(classData, X[i])
            }
        }
        
        if len(classData) == 0 {
            return fmt.Errorf("class %d has no samples", class)
        }
        
        nb.ClassLogPriors[class] = math.Log(float64(len(classData)) / float64(len(y)))
        
        nb.FeatureMeans[class] = make([]decimal.Decimal, nFeatures)
        nb.FeatureVars[class] = make([]decimal.Decimal, nFeatures)
        
        for j := 0; j < nFeatures; j++ {
            sum := decimal.Zero
            for _, row := range classData {
                sum = sum.Add(row[j])
            }
            mean := sum.Div(decimal.NewFromInt(int64(len(classData))))
            nb.FeatureMeans[class][j] = mean
            
            variance := decimal.Zero
            for _, row := range classData {
                diff := row[j].Sub(mean)
                variance = variance.Add(diff.Mul(diff))
            }
            variance = variance.Div(decimal.NewFromInt(int64(len(classData))))
            nb.FeatureVars[class][j] = variance.Add(nb.VarSmoothing)
        }
    }
    
    return nil
}

func (nb *NaiveBayes) logGaussianPDF(x, mean, variance decimal.Decimal) float64 {
    if variance.IsZero() {
        variance = nb.VarSmoothing
    }
    
    xFloat, _ := x.Float64()
    meanFloat, _ := mean.Float64()
    varFloat, _ := variance.Float64()
    
    logTwoPiVar := math.Log(2 * math.Pi * varFloat)
    diff := xFloat - meanFloat
    exponent := -(diff * diff) / (2 * varFloat)
    
    return -0.5*logTwoPiVar + exponent
}

func (nb *NaiveBayes) Predict(X [][]decimal.Decimal) []int {
    predictions := make([]int, len(X))
    
    for i, sample := range X {
        maxLogProb := math.Inf(-1)
        bestClass := nb.Classes[0]
        
        for _, class := range nb.Classes {
            logProb := nb.ClassLogPriors[class]
            
            for j, feature := range sample {
                if nb.UseLogSpace {
                    logProb += nb.logGaussianPDF(
                        feature,
                        nb.FeatureMeans[class][j],
                        nb.FeatureVars[class][j],
                    )
                }
            }
            
            if logProb > maxLogProb {
                maxLogProb = logProb
                bestClass = class
            }
        }
        
        predictions[i] = bestClass
    }
    
    return predictions
}

func (nb *NaiveBayes) PredictProba(X [][]decimal.Decimal) [][]decimal.Decimal {
    proba := make([][]decimal.Decimal, len(X))
    
    for i, sample := range X {
        logProbs := make([]float64, len(nb.Classes))
        
        for k, class := range nb.Classes {
            logProb := nb.ClassLogPriors[class]
            
            for j, feature := range sample {
                logProb += nb.logGaussianPDF(
                    feature,
                    nb.FeatureMeans[class][j],
                    nb.FeatureVars[class][j],
                )
            }
            
            logProbs[k] = logProb
        }
        
        maxLogProb := logProbs[0]
        for _, lp := range logProbs[1:] {
            if lp > maxLogProb {
                maxLogProb = lp
            }
        }
        
        sumExp := 0.0
        for _, lp := range logProbs {
            sumExp += math.Exp(lp - maxLogProb)
        }
        
        proba[i] = make([]decimal.Decimal, len(nb.Classes))
        for j, lp := range logProbs {
            prob := math.Exp(lp - maxLogProb) / sumExp
            proba[i][j] = decimal.NewFromFloat(prob)
        }
    }
    
    return proba
}

func (nb *NaiveBayes) GetClasses() []int {
    return nb.Classes
}

func (nb *NaiveBayes) Reset() {
    nb.ClassLogPriors = nil
    nb.FeatureMeans = nil
    nb.FeatureVars = nil
    nb.Classes = nil
}