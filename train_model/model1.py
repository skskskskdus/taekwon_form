import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

#하이퍼파라미터 및 설정 및 데이터셋 경로 설정
# 데이터셋 경로 설정
data = os.path.abspath("dataset")  # 절대 경로로 변환

#데이터셋 불러오기 (학습,검증 분할)
# 데이터셋 불러오기
train = image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(1080, 1980),
    batch_size=32,
    shuffle=True
)

val = image_dataset_from_directory(
    data,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(1080, 1980),
    batch_size=32,
    shuffle=True
)

"""
#클래스 이름과 클래스 수 확인
class_name=train.class_names
num_classes=len(class_name)
print("클래스 이름:", class_name)
"""

#2. EfficentNetB4 전처리 적용
preproce_input=tf.keras.applications.efficientnet.preprocess_input
train = train.map(lambda x, y: (preproce_input(x), y))
val = val.map(lambda x, y: (preproce_input(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train = train.cache().prefetch(buffer_size=AUTOTUNE)
val = val.cache().prefetch(buffer_size=AUTOTUNE)

#EfficientNetB4 모델 불러오기(사전 학습 가중치, include_top=False)
base__model=EfficientNetB4(input_shape=(1080, 1980, 3), include_top=False, weights='imagenet')
#처음에는 백본 고정
base__model.trainable=False

#4.새 분류기 구성 및 전체 모델 생성
inputs=tf.keras.Input(shape=(1080, 1980, 3))
x=base__model(inputs, training=False)
x=layers.GlobalAveragePooling2D()(x)
x=layers.Dropout(0.4)(x) #일단 학습 시켜보고 결과 따라서, 일단 EfficientNetB4의 기본 드롭아웃 비율은 0.4, 0.5로 변경해보기
outputs=layers.Dense(32, activation='softmax')(x) #32개로 변경
model=models.Model(inputs, outputs)

#5. 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

#6. 모델 컴파일: 전이 학습 단계
epoch = 10  # 초기 에포크 수
history = model.fit(train, epochs=epoch, validation_data=val)

#7. 파인튜닝 단계: 백본 일부 해제해서 미세 조정
# 백본의 모든 레이어를 훈련 가능하게 변경한 후 , 일부 초기 레이어는 다시 고정
base__model.trainable=True
fine_tune_at=int(len(base__model.layers)*0.8) # 전체 레이어의 80%를 고정

# 낮은 학습률로 재컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 10  # 파인튜닝 에포크 수
total_epochs = epoch + fine_tune_epochs

history_fine = model.fit(train, epochs=total_epochs, initial_epoch=epoch, validation_data=val)

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
# Define the missing callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath="model_checkpoint.h5", save_best_only=True)
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3)
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} ended. Logs: {logs}")

callbacks_list = [checkpoint_cb, reduce_lr_cb, EpochLogger(), tensorboard_cb]
