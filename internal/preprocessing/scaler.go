package preprocessing

import (
    "fmt"
    "math"
    "github.com/shopspring/decimal"
)

type Scaler struct {
    ScaleType string
    IsFitted  bool
    FeatureMin []decimal.Decimal
    FeatureMax []decimal.Decimal
    FeatureMean []decimal.Decimal
    FeatureStd  []decimal.Decimal
}

func NewScaler(scaleType string) *Scaler {
    return &Scaler{
        ScaleType: scaleType,
        IsFitted:  false,
    }
}

func (s *Scaler) Fit(X [][]decimal.Decimal) error {
    if len(X) == 0 {
        return fmt.Errorf("empty dataset")
    }

    nFeatures := len(X[0])
    s.FeatureMin = make([]decimal.Decimal, nFeatures)
    s.FeatureMax = make([]decimal.Decimal, nFeatures)
    s.FeatureMean = make([]decimal.Decimal, nFeatures)
    s.FeatureStd = make([]decimal.Decimal, nFeatures)

    switch s.ScaleType {
    case "minmax", "normalized":
        s.fitMinMax(X)
    case "standard", "standardized":
        s.fitStandard(X)
    case "raw", "none":
    default:
        return fmt.Errorf("unknown scale type: %s", s.ScaleType)
    }

    s.IsFitted = true
    return nil
}

func (s *Scaler) Transform(X [][]decimal.Decimal) ([][]decimal.Decimal, error) {
    if !s.IsFitted {
        return nil, fmt.Errorf("scaler must be fitted before transform")
    }

    if s.ScaleType == "raw" || s.ScaleType == "none" {
        result := make([][]decimal.Decimal, len(X))
        for i := range X {
            result[i] = make([]decimal.Decimal, len(X[i]))
            copy(result[i], X[i])
        }
        return result, nil
    }

    result := make([][]decimal.Decimal, len(X))
    for i := range X {
        result[i] = make([]decimal.Decimal, len(X[i]))
        for j := range X[i] {
            switch s.ScaleType {
            case "minmax", "normalized":
                result[i][j] = s.transformMinMax(X[i][j], j)
            case "standard", "standardized":
                result[i][j] = s.transformStandard(X[i][j], j)
            }
        }
    }

    return result, nil
}

func (s *Scaler) FitTransform(X [][]decimal.Decimal) ([][]decimal.Decimal, error) {
    if err := s.Fit(X); err != nil {
        return nil, err
    }
    return s.Transform(X)
}

func (s *Scaler) fitMinMax(X [][]decimal.Decimal) {
    nFeatures := len(X[0])

    for j := 0; j < nFeatures; j++ {
        s.FeatureMin[j] = X[0][j]
        s.FeatureMax[j] = X[0][j]

        for i := 1; i < len(X); i++ {
            if X[i][j].LessThan(s.FeatureMin[j]) {
                s.FeatureMin[j] = X[i][j]
            }
            if X[i][j].GreaterThan(s.FeatureMax[j]) {
                s.FeatureMax[j] = X[i][j]
            }
        }
    }
}

func (s *Scaler) fitStandard(X [][]decimal.Decimal) {
    nFeatures := len(X[0])
    nSamples := decimal.NewFromInt(int64(len(X)))

    for j := 0; j < nFeatures; j++ {
        sum := decimal.Zero
        for i := 0; i < len(X); i++ {
            sum = sum.Add(X[i][j])
        }
        s.FeatureMean[j] = sum.Div(nSamples)
    }

    for j := 0; j < nFeatures; j++ {
        variance := decimal.Zero
        for i := 0; i < len(X); i++ {
            diff := X[i][j].Sub(s.FeatureMean[j])
            variance = variance.Add(diff.Mul(diff))
        }
        variance = variance.Div(nSamples)

        varFloat, _ := variance.Float64()
        stdFloat := math.Sqrt(varFloat)
        s.FeatureStd[j] = decimal.NewFromFloat(stdFloat)

        if s.FeatureStd[j].IsZero() {
            s.FeatureStd[j] = decimal.NewFromInt(1)
        }
    }
}

func (s *Scaler) transformMinMax(value decimal.Decimal, featureIndex int) decimal.Decimal {
    range_ := s.FeatureMax[featureIndex].Sub(s.FeatureMin[featureIndex])
    if range_.IsZero() {
        return decimal.Zero
    }
    return value.Sub(s.FeatureMin[featureIndex]).Div(range_)
}

func (s *Scaler) transformStandard(value decimal.Decimal, featureIndex int) decimal.Decimal {
    return value.Sub(s.FeatureMean[featureIndex]).Div(s.FeatureStd[featureIndex])
}