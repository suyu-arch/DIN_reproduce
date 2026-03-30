import argparse
import datetime
from pathlib import Path#用于处理文件路径和目录，提供了面向对象的接口，方便进行路径操作和文件系统交互

import torch

from dataset import DataIteratorTorch
from model import build_model
from utils import (
    calc_auc,
    calc_rela_impr,
    count_lines,
    ensure_dir,
    estimate_steps,
    save_json,
    set_seed,
    tf_style_accuracy,
    tf_style_ctr_loss,
)


BASELINE_MODEL_NAME = "WIDE_DEEP"


def parse_args():
    #解析命令行参数，设置默认值和选项，包括训练和测试模式、数据目录、输出目录、批次大小、最大序列长度、嵌入维度、学习率、训练轮数、评估和保存的迭代次数、随机种子、设备选择、模型名称和基线模型名称等
    project_root = Path(__file__).resolve().parents[1]#设置默认数据目录为项目根目录，默认输出目录为当前文件所在目录下的 "outputs" 文件夹
    default_data_dir = project_root# / "data"
    default_output_dir = Path(__file__).resolve().parent / "outputs"

    parser = argparse.ArgumentParser(description="PyTorch DIN/Wide&Deep reproduction for this project.")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--data-dir", default=str(default_data_dir))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=100)#历史行为序列的最大长度，超过该长度的序列将被截断，较短的序列将被填充，以保证输入数据的一致性和模型的稳定训练
    parser.add_argument("--embedding-dim", type=int, default=18)
    parser.add_argument("--lr", type=float, default=1e-3)#学习率，控制模型参数更新的步长，较大的学习率可能导致训练不稳定，较小的学习率可能导致训练过慢，默认值为 0.001
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-iter", type=int, default=100)#评估的迭代次数，即每隔多少次训练迭代进行一次评估，默认值为 100，较小的 eval_iter 可以更频繁地监控模型性能，但可能增加训练时间，较大的 eval_iter 可以减少评估次数，但可能错过最佳模型的保存时机
    parser.add_argument("--save-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", default="DIN", choices=["DIN", "WIDE_DEEP"])
    parser.add_argument("--baseline-model-name", default=BASELINE_MODEL_NAME, choices=["WIDE_DEEP"])
    return parser.parse_args()


def build_paths(args):
    #根据命令行参数构建数据路径和输出路径，包括训练文件、测试文件、用户词汇表、商品词汇表、类目词汇表等，并确保输出目录和相关子目录存在
    data_dir = Path(args.data_dir)
    output_dir = ensure_dir(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")#确保输出目录下的 "checkpoints" 子目录存在，用于保存模型检查点文件
    log_dir = ensure_dir(output_dir / "logs")
    compare_dir = ensure_dir(output_dir / "comparisons")

    train_file = data_dir / "local_train_splitByUser"
    test_file = data_dir / "local_test_splitByUser"
    uid_voc = data_dir / "uid_voc.pkl"
    mid_voc = data_dir / "mid_voc.pkl"
    cat_voc = data_dir / "cat_voc.pkl"

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "ckpt_dir": ckpt_dir,
        "log_dir": log_dir,
        "compare_dir": compare_dir,
        "train_file": train_file,
        "test_file": test_file,
        "uid_voc": uid_voc,
        "mid_voc": mid_voc,
        "cat_voc": cat_voc,
    }


def make_iterator(file_path, paths, args):
    #创建数据迭代器，使用 DataIteratorTorch 类加载指定文件路径的数据，并根据用户词汇表、商品词汇表、类目词汇表进行编码，设置批次大小、最大序列长度等参数，返回一个可迭代的数据加载器对象
    return DataIteratorTorch(
        file_path,
        paths["uid_voc"],
        paths["mid_voc"],
        paths["cat_voc"],
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        sort_by_length=True,
    )


def to_device(batch, device):
    #将numpy转化成tensor，放到GPU上
    return {key: torch.as_tensor(value, device=device) for key, value in batch.items()}


def evaluate(model, file_path, paths, args, device):
    model.eval()#切换到评估模式，关闭dropout等训练时特有的行为，以确保模型在评估阶段的稳定性和一致性
    iterator = make_iterator(file_path, paths, args)

    stored_arr = []#用于存储预测概率和真实标签的列表，评估过程中会将每个样本的预测概率和真实标签添加到该列表中，以便后续计算 AUC指标
    loss_sum = 0.0
    accuracy_sum = 0.0
    nums = 0#用于统计评估过程中处理的样本数量，初始值为 0，每处理一个批次的数据，就将该批次的样本数量累加到 nums 中，以便后续计算平均损失和平均准确率

    with torch.no_grad():#在评估过程中禁用梯度计算，以节省内存和加速计算，因为评估阶段不需要进行反向传播和参数更新
        for batch in iterator:
            batch = to_device(batch, device)#将当前批次的数据转换到指定的设备上，以便模型能够在该设备上进行计算
            logits = model(batch)#模型的前向传播，输入数据批次并得到预测的 logits，logits 是模型输出的原始分数，每行对应一个样本，每列对应一个类别的分数（类别0点击和类别1未点击）
            loss = tf_style_ctr_loss(logits, batch["class_targets"])#计算当前批次的损失，使用 tf_style_ctr_loss 函数将模型的预测 logits 和真实的类别标签 batch["class_targets"] 作为输入，得到一个标量损失值，表示模型在该批次上的预测与真实标签之间的差距
            probs = torch.softmax(logits, dim=1)#将模型的预测 logits 转换为概率分布，使用 softmax 函数对 logits 进行归一化，得到每个类别的预测概率，probs 是一个二维张量，其中每行对应一个样本，每列对应一个类别的预测概率

            nums += 1#将当前批次的样本数量累加到 nums 中
            loss_sum += loss.item()#将当前批次的损失值累加到 loss_sum 中，loss.item() 用于获取损失张量的标量值，以便进行累加
            accuracy_sum += tf_style_accuracy(logits, batch["class_targets"])

            prob_1 = probs[:, 0].detach().cpu().tolist()#从预测概率张量 probs 中提取第一列（对应类别 0 的概率），使用 detach() 方法将其从计算图中分离出来，使用 cpu() 方法将其移动到 CPU 上，最后使用 tolist() 方法将其转换为一个 Python 列表，得到每个样本被预测为类别 0 的概率列表 prob_1
            target_1 = batch["click_targets"].detach().cpu().tolist()
            for p, t in zip(prob_1, target_1):
                stored_arr.append([p, float(t)])

    auc = calc_auc(stored_arr)
    return {
        "auc": auc,
        "loss": loss_sum / max(nums, 1),
        "accuracy": accuracy_sum / max(nums, 1),
    }


def checkpoint_name(model_name, suffix, seed):
    #生成模型检查点文件的名称，使用模型名称、后缀和随机种子作为组成部分，返回一个字符串格式的文件名，例如 "din_best_seed3.pt" 或 "wide_deep_step100_seed3.pt"
    return "%s_%s_seed%s.pt" % (model_name.lower(), suffix, seed)


def best_checkpoint_path(paths, model_name, seed):
    #根据模型名称和随机种子生成最佳模型检查点文件的路径，使用 checkpoint_name 函数生成文件名，并将其与检查点目录路径结合起来，返回一个 Path 对象，表示最佳模型检查点文件的完整路径
    return paths["ckpt_dir"] / checkpoint_name(model_name, "best", seed)


def save_checkpoint(path, model, optimizer, epoch, step, best_auc, args):
    #保存模型检查点文件，使用 torch.save 函数将模型的状态字典、优化器的状态字典、当前训练轮数、当前训练迭代次数、最佳 AUC 值和命令行参数等信息保存到指定的路径中，以便后续加载和恢复模型状态
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_auc": best_auc,
            "args": vars(args),
        },
        path,
    )


def log_line(log_file, text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


def maybe_load_baseline_auc(paths, args, device):
    #如果指定的基线模型名称与当前模型名称相同，则返回 None，表示不需要加载基线 AUC；
    # 否则，尝试加载基线模型的最佳检查点文件，如果文件存在，则创建基线模型实例，加载检查点文件中的模型状态字典，并使用 evaluate 函数评估基线模型在测试集上的性能，返回基线模型的 AUC 值；
    # 如果检查点文件不存在，则返回 None，表示无法加载基线 AUC
    if args.model_name == args.baseline_model_name:
        return None

    baseline_path = best_checkpoint_path(paths, args.baseline_model_name, args.seed)#baseline最好模型检查点文件路径
    if not baseline_path.exists():
        return None

    train_iterator = make_iterator(paths["train_file"], paths, args)#
    n_uid, n_mid, n_cat = train_iterator.get_n()
    baseline_model = build_model(args.baseline_model_name, n_uid, n_mid, n_cat, embedding_dim=args.embedding_dim).to(device)
    checkpoint = torch.load(baseline_path, map_location=device)
    baseline_model.load_state_dict(checkpoint["model_state_dict"])
    return evaluate(baseline_model, paths["test_file"], paths, args, device)["auc"]


def train(args):
    paths = build_paths(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    '''创建数据迭代器、获取用户数量、商品数量和类目数量、构建模型实例、创建优化器实例'''
    train_iterator = make_iterator(paths["train_file"], paths, args)
    n_uid, n_mid, n_cat = train_iterator.get_n()
    model = build_model(args.model_name, n_uid, n_mid, n_cat, embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    '''统计训练数据总数、每轮需要迭代的次数、创建实验目录、日志文件、指标文件、配置文件'''
    train_lines = count_lines(paths["train_file"])#统计总行数
    steps_per_epoch = estimate_steps(train_lines, args.batch_size)#每个训练轮需要迭代的次数
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")#获取当前时间戳
    run_name = "%s_seed%s_%s" % (args.model_name, args.seed, timestamp)#生成当前训练运行的名称，包含模型名称、随机种子和时间戳，以便区分不同的训练运行和保存相应的日志和模型检查点文件
    run_dir = ensure_dir(paths["output_dir"] / run_name)
    log_file_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.tsv"#保存指标
    config_path = run_dir / "config.json"#保存配置参数
    save_json(config_path, vars(args))

    '''保存最佳模型检查点、打开指标文件写入相关信息'''
    best_model_path = best_checkpoint_path(paths, args.model_name, args.seed)#最佳模型检查点文件路径
    metrics_file = metrics_path.open("w", encoding="utf-8")
    metrics_file.write("phase\tepoch\tstep\tauc\tloss\taccuracy\trela_impr\tlr\n")
    #写入指标文件的表头，包含阶段（训练或评估）、训练轮数、训练迭代次数、AUC值、损失值、准确率、相对提升和学习率等列，以便后续记录和分析模型的性能指标

    '''加载基线模型的 AUC 值，如果存在的话，并记录初始评估结果'''
    baseline_auc = maybe_load_baseline_auc(paths, args, device)
    best_auc = 0.0
    global_step = 0

    '''训练循环，包含每轮的训练和评估逻辑，以及模型检查点的保存'''
    with log_file_path.open("w", encoding="utf-8") as log_file:
        log_line(log_file, "device: %s" % device)
        log_line(log_file, "model_name: %s" % args.model_name)
        log_line(log_file, "baseline_model_name: %s" % args.baseline_model_name)
        log_line(log_file, "train_lines: %d" % train_lines)
        log_line(log_file, "steps_per_epoch: %d" % steps_per_epoch)
        if baseline_auc is not None:
            log_line(log_file, "baseline_auc(%s): %.6f" % (args.baseline_model_name, baseline_auc))
        
        '''初始评估'''
        initial_eval = evaluate(model, paths["test_file"], paths, args, device)
        initial_rela_impr = calc_rela_impr(initial_eval["auc"], baseline_auc)
        #计算初始评估的相对提升，如果 baseline_auc 不为 None，则使用 calc_rela_impr 函数计算当前模型的 AUC 相对于基线模型 AUC 的相对提升百分比，存储在 initial_rela_impr 变量中
        log_line(
            log_file,
            "initial ---- test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f%s"
            % (
                initial_eval["auc"],
                initial_eval["loss"],
                initial_eval["accuracy"],
                "" if initial_rela_impr is None else " ---- RelaImpr(%s): %.2f%%" % (args.baseline_model_name, initial_rela_impr),
            ),
        )
        metrics_file.write("eval\t0\t0\t%.6f\t%.6f\t%.6f\t%s\t%.8f\n" % (
            initial_eval["auc"],
            initial_eval["loss"],
            initial_eval["accuracy"],
            "" if initial_rela_impr is None else "%.6f" % initial_rela_impr,
            optimizer.param_groups[0]["lr"],
        ))
        '''保存初始评估结果的模型检查点，如果初始评估的 AUC 值优于当前最佳 AUC，则保存模型检查点文件，并更新最佳 AUC'''
        if initial_eval["auc"] > best_auc:
            best_auc = initial_eval["auc"]
            save_checkpoint(best_model_path, model, optimizer, 0, 0, best_auc, args)

        '''训练循环，包含每轮的训练和评估逻辑，以及模型检查点的保存'''
        for epoch in range(1, args.epochs + 1):
            model.train()
            iterator = make_iterator(paths["train_file"], paths, args)
            window_loss = 0.0#用于累积当前评估窗口内的损失值，初始值为 0.0，每处理一个批次的数据，就将该批次的损失值累加到 window_loss 中，以便后续计算评估窗口内的平均损失
            window_accuracy = 0.0
            window_steps = 0
            '''每个batch的训练逻辑，包括前向传播、损失计算、反向传播和参数更新，以及评估窗口内的指标累积和定期评估模型性能'''
            for batch in iterator:
                global_step += 1
                window_steps += 1

                batch = to_device(batch, device)
                optimizer.zero_grad()#在每个训练迭代开始时，使用 optimizer.zero_grad() 方法将模型参数的梯度清零
                logits = model(batch)
                loss = tf_style_ctr_loss(logits, batch["class_targets"])
                loss.backward()#在计算完当前批次的损失后，使用 loss.backward() 方法进行反向传播，计算模型参数的梯度，以便后续进行参数更新
                optimizer.step()#在完成反向传播后，使用 optimizer.step() 方法更新模型参数

                window_loss += loss.item()#将当前批次的损失值累加到 window_loss 中，loss.item() 用于获取损失张量的标量值，以便进行累加
                window_accuracy += tf_style_accuracy(logits, batch["class_targets"])

                '''每eval_iter进行一次评估，计算评估窗口内的平均损失和平均准确率，并使用 evaluate 函数评估模型在测试集上的性能，记录评估结果和相对提升，并保存最佳模型检查点文件'''
                if global_step % args.eval_iter == 0:
                    train_loss = window_loss / max(window_steps, 1)
                    train_acc = window_accuracy / max(window_steps, 1)
                    log_line(
                        log_file,
                        "epoch: %d ---- iter: %d ---- train_loss: %.4f ---- train_accuracy: %.4f"
                        % (epoch, global_step, train_loss, train_acc),
                    )
                    metrics_file.write("train\t%d\t%d\t%.6f\t%.6f\t%.6f\t\t%.8f\n" % (
                        epoch, global_step, 0.0, train_loss, train_acc, optimizer.param_groups[0]["lr"]
                    ))

                    eval_result = evaluate(model, paths["test_file"], paths, args, device)
                    rela_impr = calc_rela_impr(eval_result["auc"], baseline_auc)
                    #计算当前评估的相对提升，如果 baseline_auc 不为 None，则使用 calc_rela_impr 函数计算当前模型的 AUC 相对于基线模型 AUC 的相对提升百分比，存储在 rela_impr 变量中
                    log_line(
                        log_file,
                        "epoch: %d ---- iter: %d ---- test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f%s"
                        % (
                            epoch,
                            global_step,
                            eval_result["auc"],
                            eval_result["loss"],
                            eval_result["accuracy"],
                            "" if rela_impr is None else " ---- RelaImpr(%s): %.2f%%" % (args.baseline_model_name, rela_impr),
                        ),
                    )
                    metrics_file.write("eval\t%d\t%d\t%.6f\t%.6f\t%.6f\t%s\t%.8f\n" % (
                        epoch,
                        global_step,
                        eval_result["auc"],
                        eval_result["loss"],
                        eval_result["accuracy"],
                        "" if rela_impr is None else "%.6f" % rela_impr,
                        optimizer.param_groups[0]["lr"],
                    ))
                    metrics_file.flush()

                    if eval_result["auc"] > best_auc:
                        best_auc = eval_result["auc"]
                        save_checkpoint(best_model_path, model, optimizer, epoch, global_step, best_auc, args)

                    window_loss = 0.0#在每次评估后，重置评估窗口内的损失累积值、准确率累积值和步骤计数器，以便开始新的评估窗口的指标累积
                    window_accuracy = 0.0
                    window_steps = 0

                # if global_step % args.save_iter == 0:
                #     ckpt_path = paths["ckpt_dir"] / checkpoint_name(args.model_name, "step%d" % global_step, args.seed)
                #     save_checkpoint(ckpt_path, model, optimizer, epoch, global_step, best_auc, args)
                #     log_line(log_file, "saved checkpoint: %s" % ckpt_path.name)
            #每个训练轮结束后，学习率衰减为之前的一半
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5

        log_line(log_file, "best_auc: %.6f" % best_auc)
        log_line(log_file, "best_model: %s" % best_model_path)

    metrics_file.close()


def test(args):
    paths = build_paths(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    train_iterator = make_iterator(paths["train_file"], paths, args)
    n_uid, n_mid, n_cat = train_iterator.get_n()
    model = build_model(args.model_name, n_uid, n_mid, n_cat, embedding_dim=args.embedding_dim).to(device)

    best_model_path = best_checkpoint_path(paths, args.model_name, args.seed)
    if not best_model_path.exists():
        raise FileNotFoundError("Best checkpoint not found: %s" % best_model_path)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    baseline_auc = maybe_load_baseline_auc(paths, args, device)
    result = evaluate(model, paths["test_file"], paths, args, device)
    rela_impr = calc_rela_impr(result["auc"], baseline_auc)

    text = "test_auc: %.4f ---- test_loss: %.4f ---- test_accuracy: %.4f" % (
        result["auc"], result["loss"], result["accuracy"]
    )
    if rela_impr is not None:
        text += " ---- RelaImpr(%s): %.2f%%" % (args.baseline_model_name, rela_impr)
    print(text)


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.mode == "train":
        train(parsed_args)
    else:
        test(parsed_args)
