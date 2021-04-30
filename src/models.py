model_attributes = {
    'bert-base-uncased': {
        'feature_type': 'text'
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
    'densenet121': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    }
}
