### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 0  # Số seed (Ban tổ chức sẽ công bố & thay đổi vào lúc chấm)
# Đường dẫn đến thư mục train
# (đúng theo cấu trúc gồm 4 thư mục cho 4 classes của ban tổ chức)
TRAIN_DATA_DIR_PATH = 'data/train'
# Đường dẫn đến thư mục test
TEST_DATA_DIR_PATH = 'data/test'
### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
# Lưu ý: các thư viện & phiên bản cài đặt vui lòng để trong requirements.txt
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###
# Seeding nhằm đảm bảo kết quả sẽ cố định
# và không ngẫu nhiên ở các lần chạy khác nhau
# Set seed for random
random.seed(SEED)
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Set seed for tensorflow
# tf.random.set_seed(SEED)
### END: SEEDING EVERYTHING - KHÔNG THAY ĐỔI ###

# START: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###
import libs.lib1
import libs.lib2
### END: IMPORT CÁC THƯ VIỆN CUSTOM, MODEL, v.v. riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
# Model sẽ được train bằng cac ảnh ở [TRAIN_DATA_DIR_PATH]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {
    "bào ngư xám + trắng": "1",
    "Đùi gà Baby (cắt ngắn)": "2",
    "linh chi trắng": "3",
    "nấm mỡ": "0"
}

train_loader, val_loader, submit_loader, class_names,idx_to_class = libs.lib2.get_dataloaders(
    train_path=TRAIN_DATA_DIR_PATH, 
    test_path=TEST_DATA_DIR_PATH, 
    label_map=label_map,
    batch_size=32
    )

model = libs.lib1.EfficientNetClassifier(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.001) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Số lượng tham số được huấn luyện: {trainable_params}")

libs.lib2.train(
    device=device, 
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader, 
    epochs=50, 
    optimizer=optimizer, 
    criterion=criterion, 
    scheduler=scheduler, 
    idx_to_class=idx_to_class
)

### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###


### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
# Kết quả dự đoán của mô hình cho tập dữ liệu các ảnh ở [TEST_DATA_DIR_PATH]
# sẽ lưu vào file "output/results.csv"
# Cấu trúc gồm 2 cột: image_name và label: (encoded: 0, 1, 2, 3)
# image_name,label
# image1.jpg,0
# image2.jpg,1
# image3.jpg,2
# image4.jpg,3

results = libs.lib2.create_submission(
    model=model,
    dataloader=submit_loader,
    idx_to_class=idx_to_class,
    device=device,
    path="output/results.csv"
)
### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###
