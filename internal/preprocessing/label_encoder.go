package preprocessing

import (
    "encoding/gob"
    "fmt"
    "os"
)

type LabelEncoder struct {
    ClassToInt map[string]int
    IntToClass map[int]string
    IsFitted   bool
}

func NewLabelEncoder() *LabelEncoder {
    return &LabelEncoder{
        ClassToInt: make(map[string]int),
        IntToClass: make(map[int]string),
        IsFitted:   false,
    }
}

func (le *LabelEncoder) Fit(labels []string) {
    le.ClassToInt = make(map[string]int)
    le.IntToClass = make(map[int]string)
    
    uniqueLabels := make(map[string]bool)
    for _, label := range labels {
        uniqueLabels[label] = true
    }
    
    idx := 0
    for label := range uniqueLabels {
        le.ClassToInt[label] = idx
        le.IntToClass[idx] = label
        idx++
    }
    
    le.IsFitted = true
}

func (le *LabelEncoder) Transform(labels []string) ([]int, error) {
    if !le.IsFitted {
        return nil, fmt.Errorf("LabelEncoder must be fitted before transform")
    }
    
    result := make([]int, len(labels))
    for i, label := range labels {
        if val, ok := le.ClassToInt[label]; ok {
            result[i] = val
        } else {
            return nil, fmt.Errorf("unknown label: %s", label)
        }
    }
    
    return result, nil
}

func (le *LabelEncoder) FitTransform(labels []string) ([]int, error) {
    le.Fit(labels)
    return le.Transform(labels)
}

func (le *LabelEncoder) InverseTransform(encoded []int) ([]string, error) {
    if !le.IsFitted {
        return nil, fmt.Errorf("LabelEncoder must be fitted before inverse transform")
    }
    
    result := make([]string, len(encoded))
    for i, val := range encoded {
        if label, ok := le.IntToClass[val]; ok {
            result[i] = label
        } else {
            return nil, fmt.Errorf("unknown encoding: %d", val)
        }
    }
    
    return result, nil
}

func (le *LabelEncoder) Save(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    encoder := gob.NewEncoder(file)
    return encoder.Encode(le)
}

func (le *LabelEncoder) Load(filename string) error {
    file, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    decoder := gob.NewDecoder(file)
    return decoder.Decode(le)
}