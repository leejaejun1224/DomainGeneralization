import json
import math
import argparse
import matplotlib.pyplot as plt

def plot_loss_graph(loss_dict, output_image_path):

    epochs = []
    train_losses = []
    val_losses = []
    train_pseudo_losses = []
    val_pseudo_losses = []

    epoch_keys = [k for k in loss_dict.keys() if k.startswith("epoch")]
    sorted_keys = sorted(epoch_keys, key=lambda k: int(k.split('_')[1]))

    
    for key in sorted_keys:
        # "epoch_1" -> 1 추출
        epoch_num = int(key.split('_')[1])
        epochs.append(epoch_num)

        # 각 epoch의 데이터 가져오기
        epoch_data = loss_dict[key]
        train_loss = epoch_data.get("train_loss")
        val_loss = epoch_data.get("val_loss")
        train_pseudo_loss = epoch_data.get("train_pseudo_loss")
        val_pseudo_loss = epoch_data.get("val_pseudo_loss")

        # 문자열 "NaN"으로 저장된 경우 처리 (대소문자 구분 없이)
        if isinstance(train_loss, str) and train_loss.lower() == "nan":
            train_loss = float('nan')
        if isinstance(val_loss, str) and val_loss.lower() == "nan":
            val_loss = float('nan')
        if isinstance(train_pseudo_loss, str) and train_pseudo_loss.lower() == "nan":
            train_pseudo_loss = float('nan')
        if isinstance(val_pseudo_loss, str) and val_pseudo_loss.lower() == "nan":
            val_pseudo_loss = float('nan')

        # 혹시 float('nan')이 이미 들어있는 경우 math.isnan()로 처리 가능 (여기서는 그냥 그대로 추가)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_pseudo_losses.append(train_pseudo_loss)
        val_pseudo_losses.append(val_pseudo_loss)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', linestyle='-')
    plt.plot(epochs, train_pseudo_losses, label='Train Pseudo Loss', marker='^', linestyle='-')
    plt.plot(epochs, val_pseudo_losses, label='Validation Pseudo Loss', marker='^', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.grid(True)

    # 이미지 파일로 저장
    plt.savefig(output_image_path)
    plt.close()

def plot_true_ratio(loss_dict, output_image_path):

    epochs = []
    train_true_ratios = []
    val_true_ratios = []

    epoch_keys = [k for k in loss_dict.keys() if k.startswith("epoch")]
    sorted_keys = sorted(epoch_keys, key=lambda k: int(k.split('_')[1]))

    
    for key in sorted_keys:
        # "epoch_1" -> 1 추출
        epoch_num = int(key.split('_')[1])
        epochs.append(epoch_num)

        # 각 epoch의 데이터 가져오기
        epoch_data = loss_dict[key]
        train_true_ratio = epoch_data.get("true_ratio_train")
        val_true_ratio = epoch_data.get("true_ratio_val")

        # 문자열 "NaN"으로 저장된 경우 처리 (대소문자 구분 없이)
        if isinstance(train_true_ratio, str) and train_true_ratio.lower() == "nan":
            train_true_ratio = float('nan')
        if isinstance(val_true_ratio, str) and val_true_ratio.lower() == "nan":
            val_true_ratio = float('nan')

        # 혹시 float('nan')이 이미 들어있는 경우 math.isnan()로 처리 가능 (여기서는 그냥 그대로 추가)
        train_true_ratios.append(train_true_ratio)
        val_true_ratios.append(val_true_ratio)

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_true_ratios, label='Train True Ratio', marker='o', linestyle='-')
    plt.plot(epochs, val_true_ratios, label='Validation True Ratio', marker='s', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('True Ratio')
    plt.title('True Ratio vs Epoch')
    plt.legend()
    plt.grid(True)

    # 이미지 파일로 저장
    plt.savefig(output_image_path)
    plt.close()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--json_file', type=str, required=True)
#     parser.add_argument('--output_image', type=str, required=True)
#     args = parser.parse_args()
#     plot_loss_graph(args.json_file, args.output_image)