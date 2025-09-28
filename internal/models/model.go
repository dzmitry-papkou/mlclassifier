package models

import (
    "github.com/shopspring/decimal"
)

type Model interface {
    Fit(X [][]decimal.Decimal, y []int) error
    Predict(X [][]decimal.Decimal) []int
    PredictProba(X [][]decimal.Decimal) [][]decimal.Decimal
    GetType() string
    GetName() string
    GetParams() map[string]any
    GetClasses() []int
    Reset()
}

type BaseModel struct {
    Name    string
    Params  map[string]any
    Classes []int
}

func (bm *BaseModel) GetType() string {
    return bm.Name
}

func (bm *BaseModel) GetName() string {
    return bm.Name
}

func (bm *BaseModel) GetParams() map[string]any {
    return bm.Params
}

func ExtractClasses(y []int) []int {
    classMap := make(map[int]bool)
    for _, label := range y {
        classMap[label] = true
    }

    classes := make([]int, 0, len(classMap))
    for class := range classMap {
        classes = append(classes, class)
    }

    return classes
}


