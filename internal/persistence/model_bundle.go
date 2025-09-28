package persistence

import (
    "encoding/gob"
    "fmt"
    "os"
    "time"
    
    "mlclassifier/internal/models"
    "mlclassifier/internal/preprocessing"
)

type ModelBundle struct {
    Model         models.Model
    Scaler        *preprocessing.Scaler
    LabelEncoder  *preprocessing.LabelEncoder
    Metadata      BundleMetadata
    CreatedAt     time.Time
}

type BundleMetadata struct {
    ModelName      string
    Dataset        string
    Accuracy       float64
    Precision      float64
    Recall         float64
    F1Score        float64
    TrainingTime   time.Duration
    Features       []string
    Classes        []string
    Parameters     map[string]any
}

func NewModelBundle(model models.Model) *ModelBundle {
    return &ModelBundle{
        Model:     model,
        CreatedAt: time.Now(),
        Metadata: BundleMetadata{
            ModelName:  model.GetName(),
            Parameters: model.GetParams(),
        },
    }
}

func (mb *ModelBundle) Save(filename string) error {
    gob.Register(&models.KNN{})
    gob.Register(&models.DecisionTree{})
    gob.Register(&models.RandomForest{})
    gob.Register(&models.NaiveBayes{})
    gob.Register(&models.TreeNode{})
    
    file, err := os.Create(filename)
    if err != nil {
        return fmt.Errorf("failed to create file: %w", err)
    }
    defer file.Close()
    
    encoder := gob.NewEncoder(file)
    if err := encoder.Encode(mb); err != nil {
        return fmt.Errorf("failed to encode bundle: %w", err)
    }
    
    return nil
}

func LoadModelBundle(filename string) (*ModelBundle, error) {
    gob.Register(&models.KNN{})
    gob.Register(&models.DecisionTree{})
    gob.Register(&models.RandomForest{})
    gob.Register(&models.NaiveBayes{})
    gob.Register(&models.TreeNode{})
    
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("failed to open file: %w", err)
    }
    defer file.Close()
    
    var bundle ModelBundle
    decoder := gob.NewDecoder(file)
    if err := decoder.Decode(&bundle); err != nil {
        return nil, fmt.Errorf("failed to decode bundle: %w", err)
    }
    
    return &bundle, nil
}

func (mb *ModelBundle) SaveMetadata(filename string) error {
    file, err := os.Create(filename)
    if err != nil {
        return err
    }
    defer file.Close()
    
    fmt.Fprintf(file, "Model: %s\n", mb.Metadata.ModelName)
    fmt.Fprintf(file, "Dataset: %s\n", mb.Metadata.Dataset)
    fmt.Fprintf(file, "Created: %s\n", mb.CreatedAt.Format(time.RFC3339))
    fmt.Fprintf(file, "Accuracy: %.4f\n", mb.Metadata.Accuracy)
    fmt.Fprintf(file, "Precision: %.4f\n", mb.Metadata.Precision)
    fmt.Fprintf(file, "Recall: %.4f\n", mb.Metadata.Recall)
    fmt.Fprintf(file, "F1 Score: %.4f\n", mb.Metadata.F1Score)
    fmt.Fprintf(file, "Training Time: %v\n", mb.Metadata.TrainingTime)
    
    return nil
}