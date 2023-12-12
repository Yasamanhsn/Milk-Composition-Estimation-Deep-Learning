from TrainNetwork.CustomLoss import custom_loss
from tensorflow.keras.optimizers import Adam


def build_model(model_name,is_use_custom_loss=True):
    """
    Builds various neural network models for regression tasks on image data.
    All model name:
    simple_cnn
    efficientnet_b0,
    efficientnet_b7, efficientnet_b7_attention
    efficientnet_v2s, efficientnet_v2s_attention
    mobilenet_v2, hybrid_mobilenet_v2_attention,
    mobilenet_v3, hybrid_mobilenet_v3_attention
    inception_resnet_v2, hybrid_inception_resnet_v2_attention
    vgg16,hybrid_vgg16_attention

    Parameters:
    model_name (str): Name of the desired model architecture.
    is_use_custom_loss (bool): Flag indicating whether to use custom loss functions. Default is True.

    Returns:
    tf.keras.Model: Constructed neural network model based on the specified model_name.
    """

    lr = 0.0001
    opt = Adam(learning_rate=lr)
    if is_use_custom_loss:
        loss = custom_loss  # Custom loss function
    else:
        loss = 'mse'  # Default Mean Squared Error loss



    if model_name == "mobilenet_v2":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import MobileNetV2
        import tensorflow as tf

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)  # 3 output units

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name=="simple_cnn":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention
        from tensorflow.keras.applications import MobileNetV2
        import tensorflow as tf

        # Define the MobileNetV2 base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers to prevent them from being updated during training
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features from the base model
        features = base_model.output

        # Apply global average pooling to reduce the feature maps to a single vector
        pooled_features = GlobalAveragePooling2D()(features)

        # Pass the pooled features through a fully connected layer with ReLU activation
        dense_features = Dense(32, activation='relu')(pooled_features)

        # Apply an attention layer to focus on the most relevant features
        attention_weights = Attention()([dense_features, dense_features])

        # Combine the attention weights with the dense features
        context_vector = tf.keras.layers.Concatenate()([dense_features, attention_weights * dense_features])

        # Add a final fully connected layer for the regression task
        predictions = Dense(3)(context_vector)  # 3 output units for regression

        # Define the model inputs and outputs
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model for training
        model.compile(optimizer=opt, loss=loss, metrics=['mse','mae'])

    elif model_name=='mobilenet_v3':
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import MobileNetV3Large
        import tensorflow as tf

        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    elif model_name == "hybrid_mobilenet_v3_attention":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention
        from tensorflow.keras.applications import MobileNetV3Large
        import tensorflow as tf

        base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers to prevent them from being updated during training
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features from the base model
        features = base_model.output

        # Apply global average pooling to reduce the feature maps to a single vector
        pooled_features = GlobalAveragePooling2D()(features)

        # Pass the pooled features through a fully connected layer with ReLU activation
        dense_features = Dense(128, activation='relu')(pooled_features)

        # Apply an attention layer to focus on the most relevant features
        attention_weights = Attention()([dense_features, dense_features])

        # Combine the attention weights with the dense features
        context_vector = tf.keras.layers.Concatenate()([dense_features, attention_weights * dense_features])

        # Add a final fully connected layer for the regression task
        predictions = Dense(3)(context_vector)  # 3 output units for regression

        # Define the model inputs and outputs
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model for training
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    elif model_name == 'efficientnet_v2s':
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

        # Build EfficientNetV2S base
        base_model = keras.applications.EfficientNetV2S(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        # Freeze base model
        base_model.trainable = False

        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3, name="pred")(x)

        # Create model
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == 'efficientnet_v2s_attention':
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention

        # Build EfficientNetV2S base
        base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
        )

        # Freeze base model layers to prevent them from being updated during training
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features from the base model
        features = base_model.output

        # Apply global average pooling to reduce the feature maps to a single vector
        pooled_features = GlobalAveragePooling2D()(features)

        # Pass the pooled features through a fully connected layer with ReLU activation
        dense_features = Dense(64, activation='relu')(pooled_features)

        # Apply an attention layer to focus on the most relevant features
        attention_weights = Attention()([dense_features, dense_features])

        # Combine the attention weights with the dense features
        context_vector = keras.layers.Concatenate()([dense_features, attention_weights * dense_features])

        # Add a final fully connected layer for the regression task
        predictions = Dense(3)(context_vector)  # 3 output units for regression

        # Define the model inputs and outputs
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == 'efficientnet_b7_attention':
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention

        # Build EfficientNetB7 base
        base_model = keras.applications.EfficientNetB7(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        # Freeze base model
        base_model.trainable = False

        # Get base outputs
        features = base_model.output

        # Pool features
        pooled_features = GlobalAveragePooling2D()(features)

        # Dense layer
        dense_features = Dense(64, activation='relu')(pooled_features)

        # Apply attention layer
        attention_weights = Attention()([dense_features, dense_features])

        # Concatenate attention weighted features
        context_vector = keras.layers.Concatenate()([dense_features, attention_weights])

        # Add regression head
        predictions = Dense(3)(context_vector)

        # Create model
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == 'efficientnet_b7':
        from tensorflow import keras
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

        # Build EfficientNetB7 base
        base_model = keras.applications.EfficientNetB7(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        # Freeze base model
        base_model.trainable = False

        # Add pooling and dense layers on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)

        # Create model
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == 'efficientnet_b0':
        from tensorflow import keras
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

        # Build EfficientNetB0 base
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )

        # Freeze base model
        base_model.trainable = False

        # Add pooling and dense layers on top
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)

        # Create model
        model = keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == "inception_resnet_v2":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import InceptionResNetV2
        import tensorflow as tf

        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)  # 3 output units

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name=="hybrid_inception_resnet_v2_attention":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention
        from tensorflow.keras.applications import MobileNetV2
        import tensorflow as tf

        # Define the MobileNetV2 base model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers to prevent them from being updated during training
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features from the base model
        features = base_model.output

        # Apply global average pooling to reduce the feature maps to a single vector
        pooled_features = GlobalAveragePooling2D()(features)

        # Pass the pooled features through a fully connected layer with ReLU activation
        dense_features = Dense(64, activation='relu')(pooled_features)

        # Apply an attention layer to focus on the most relevant features
        attention_weights = Attention()([dense_features, dense_features])

        # Combine the attention weights with the dense features
        context_vector = tf.keras.layers.Concatenate()([dense_features, attention_weights * dense_features])

        # Add a final fully connected layer for the regression task
        predictions = Dense(3)(context_vector)  # 3 output units for regression

        # Define the model inputs and outputs
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model for training
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == "vgg16":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import VGG16
        import tensorflow as tf

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)  # 3 output units

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == "vgg19":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import VGG19
        import tensorflow as tf

        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)  # 3 output units

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name == "resnet50":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras.applications import ResNet50
        import tensorflow as tf

        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(3)(x)  # 3 output units

        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    # Model creation based on the provided model_name
    elif model_name == "hybrid_mobilenet_v2_attention":
        # Simple Convolutional Neural Network
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

        model = Sequential()

        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
        model.add(Conv2D(16, (3, 3), activation='relu'))

        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))

        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))

        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(Conv2D(128, (3, 3), activation='relu'))

        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        model.add(Dense(64, activation='relu'))
        model.add(Dense(3))  # Output layer with 3 nodes for regression values

        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    elif model_name=="hybrid_vgg16_attention":
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Attention
        from tensorflow.keras.applications import VGG16
        import tensorflow as tf

        # Define the vgg16 base model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model layers to prevent them from being updated during training
        for layer in base_model.layers:
            layer.trainable = False

        # Extract features from the base model
        features = base_model.output

        # Apply global average pooling to reduce the feature maps to a single vector
        pooled_features = GlobalAveragePooling2D()(features)

        # Pass the pooled features through a fully connected layer with ReLU activation
        dense_features = Dense(64, activation='relu')(pooled_features)

        # Apply an attention layer to focus on the most relevant features
        attention_weights = Attention()([dense_features, dense_features])

        # Combine the attention weights with the dense features
        context_vector = tf.keras.layers.Concatenate()([dense_features, attention_weights * dense_features])

        # Add a final fully connected layer for the regression task
        predictions = Dense(3)(context_vector)  # 3 output units for regression

        # Define the model inputs and outputs
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        # Compile the model for training
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])

    model.summary()

    return model