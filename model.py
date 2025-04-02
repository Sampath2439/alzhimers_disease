import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, Conv2D, MaxPooling2D, Flatten, Reshape, TimeDistributed

def create_cnn_feature_extractor(input_shape=(128, 128, 3)):
    """Create a CNN model for feature extraction from MRI images"""
    
    # Create a CNN feature extractor based on a smaller custom architecture
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
    ])
    
    return model

def create_cnn_lstm_model(input_shape=(128, 128, 3), num_classes=4, bidirectional=False):
    """
    Create a CNN-LSTM model for Alzheimer's disease classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classification categories
        bidirectional: Whether to use Bidirectional LSTM (True) or standard LSTM (False)
    """
    
    # CNN feature extractor
    cnn_model = create_cnn_feature_extractor(input_shape)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Extract features using CNN
    features = cnn_model(inputs)
    
    # Reshape for LSTM input (treat spatial dimensions as time steps)
    # After CNN, the feature map should be (8, 8, 256)
    # Reshape to (64, 256) where 64 is the sequence length and 256 is the feature dimension
    reshaped = Reshape((-1, 256))(features)
    
    # LSTM layers - either standard or bidirectional based on parameter
    if bidirectional:
        # Bidirectional LSTM layers
        lstm_out = Bidirectional(LSTM(128, return_sequences=True))(reshaped)
        lstm_out = Bidirectional(LSTM(64))(lstm_out)
    else:
        # Standard LSTM layers
        lstm_out = LSTM(128, return_sequences=True)(reshaped)
        lstm_out = LSTM(64)(lstm_out)
    
    # Dense layers for classification
    dense = Dense(128, activation='relu')(lstm_out)
    dropout = Dropout(0.5)(dense)
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create specific models with standard LSTM or BiLSTM
def create_cnn_bilstm_model(input_shape=(128, 128, 3), num_classes=4):
    """Create a CNN-BiLSTM model for Alzheimer's disease classification"""
    return create_cnn_lstm_model(input_shape, num_classes, bidirectional=True)

def create_sequential_cnn_lstm_model(input_shape=(128, 128, 3), num_classes=4, bidirectional=True):
    """
    Alternative implementation of CNN-LSTM model using TimeDistributed
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of classification categories
        bidirectional: Whether to use Bidirectional LSTM (True) or standard LSTM (False)
    """
    
    # First, create a model that extracts features using CNN
    feature_extractor = Sequential([
        # Create a CNN for feature extraction
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten()
    ])
    
    # Now create a sequence model that uses TimeDistributed to apply the CNN to each frame
    # For a single image input, we would reshape to (1, h, w, c) to treat it as a sequence of length 1
    model_layers = [
        # Reshape input to (sequence_length, height, width, channels)
        # For a single image, sequence_length would be 1
        Reshape((1,) + input_shape, input_shape=input_shape),
        
        # Apply CNN to each time step
        TimeDistributed(feature_extractor),
    ]
    
    # Add either standard or bidirectional LSTM layers
    if bidirectional:
        # Bidirectional LSTM layers for processing in both directions
        model_layers.extend([
            Bidirectional(LSTM(256, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(128)),
            Dropout(0.5),
        ])
    else:
        # Standard LSTM layers for sequential processing
        model_layers.extend([
            LSTM(256, return_sequences=True),
            Dropout(0.5),
            LSTM(128),
            Dropout(0.5),
        ])
    
    # Add final dense layers for classification
    model_layers.extend([
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Create the model with all layers
    model = Sequential(model_layers)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Helper functions to create specific model variants
def create_sequential_cnn_bilstm_model(input_shape=(128, 128, 3), num_classes=4):
    """Create a Sequential CNN-BiLSTM model for Alzheimer's disease classification"""
    return create_sequential_cnn_lstm_model(input_shape, num_classes, bidirectional=True)
