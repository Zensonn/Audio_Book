import regex
import torch
import json
import unicodedata
import re
import os
import numpy as np
import streamlit as st
from types import SimpleNamespace
from models import DurationNet, SynthesizerTrn

torch.manual_seed(42)

config_file = "config/config.json"
duration_model_path = "duration_model/vbx_duration_model.pth"
lightspeed_model_path = "model_Tacotron/gen_141k.pth"
phone_set_file = "vbx_phone_set.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tệp cấu hình
with open(config_file, "rb") as f:
    hps = json.load(f, object_hook=lambda x: SimpleNamespace(**x))

# Tệp JSON chứa tập hợp cấu 
with open(phone_set_file, "r") as f:
    phone_set = json.load(f)
    
assert phone_set[0][1:-1] == "SEP"
assert "sil" in phone_set
sil_idx = phone_set.index("sil")

space_re = regex.compile(r"\s+")
number_re = regex.compile("([0-9]+)")
digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
num_re = regex.compile(r"([0-9.,]*[0-9])")
alphabet = "aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵbcdđghklmnpqrstvx"
keep_text_and_num_re = regex.compile(rf"[^\s{alphabet}.,0-9]")
keep_text_re = regex.compile(rf"[^\s{alphabet}]")


def read_number(num: str) -> str:
    if len(num) == 1:
        return digits[int(num)]
    elif len(num) == 2 and num.isdigit():
        n = int(num)
        end = digits[n % 10]
        if n == 10:
            return "mười"
        if n % 10 == 5:
            end = "lăm"
        if n % 10 == 0:
            return digits[n // 10] + " mươi"
        elif n < 20:
            return "mười " + end
        else:
            if n % 10 == 1:
                end = "mốt"
            return digits[n // 10] + " mươi " + end
    elif len(num) == 3 and num.isdigit():
        n = int(num)
        if n % 100 == 0:
            return digits[n // 100] + " trăm"
        elif num[1] == "0":
            return digits[n // 100] + " trăm lẻ " + digits[n % 100]
        else:
            return digits[n // 100] + " trăm " + read_number(num[1:])
    elif len(num) >= 4 and len(num) <= 6 and num.isdigit():
        n = int(num)
        n1 = n // 1000
        return read_number(str(n1)) + " ngàn " + read_number(num[-3:])
    elif "," in num:
        n1, n2 = num.split(",")
        return read_number(n1) + " phẩy " + read_number(n2)
    elif "." in num:
        parts = num.split(".")
        if len(parts) == 2:
            if parts[1] == "000":
                return read_number(parts[0]) + " ngàn"
            elif parts[1].startswith("00"):
                end = digits[int(parts[1][2:])]
                return read_number(parts[0]) + " ngàn lẻ " + end
            else:
                return read_number(parts[0]) + " ngàn " + read_number(parts[1])
        elif len(parts) == 3:
            return (
                read_number(parts[0])
                + " triệu "
                + read_number(parts[1])
                + " ngàn "
                + read_number(parts[2])
            )
    return num


def text_to_phone_idx(text):
    # Chuyển đổi chuỗi ký tự thành chữ thường
    text = text.lower()
    # Chuẩn hóa chuỗi Unicode
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace(";", " ; ")
    text = text.replace(":", " : ")
    text = text.replace("!", " ! ")
    text = text.replace("?", " ? ")
    text = text.replace("(", " ( ")

    text = num_re.sub(r" \1 ", text)
    words = text.split()
    words = [read_number(w) if num_re.fullmatch(w) else w for w in words]
    text = " ".join(words)

    # loại bỏ các khoảng trắng dư thừa
    text = re.sub(r"\s+", " ", text)
    # loại bỏ khoảng trắng ở đầu và cuối
    text = text.strip()
    # chuyển đổi từ thành chỉ số 
    tokens = []
    for c in text:
        # nếu c là ',' hoặc '.', thêm âm vị <sil> thời gian yên lặng" (<silence>)
        if c in ":,.!?;(":
            tokens.append(sil_idx)
        elif c in phone_set:
            tokens.append(phone_set.index(c))
        elif c == " ":
            # thêm âm vị <sep>  "separator" (phân tách) 
            tokens.append(0)
    if tokens[0] != sil_idx:
        # chèn âm vị <sil> ở đầu, yên lặng (silence) 
        tokens = [sil_idx, 0] + tokens
    if tokens[-1] != sil_idx:
        tokens = tokens + [0, sil_idx]
    return tokens


def text_to_speech(duration_net, generator, text):
    # Cản trở văn bản quá dài
    if len(text) > 500:
        text = text[:500]

    phone_idx = text_to_phone_idx(text)
    batch = {
        "phone_idx": np.array([phone_idx]),
        "phone_length": np.array([len(phone_idx)]),
    }

    # dự đoán thời gian của âm vị
    phone_length = torch.from_numpy(batch["phone_length"].copy()).long().to(device)
    phone_idx = torch.from_numpy(batch["phone_idx"].copy()).long().to(device)
    with torch.inference_mode():
        phone_duration = duration_net(phone_idx, phone_length)[:, :, 0] * 1000
    phone_duration = torch.where(
        phone_idx == sil_idx, torch.clamp_min(phone_duration, 200), phone_duration
    )
    phone_duration = torch.where(phone_idx == 0, 0, phone_duration)

    # tạo ra hình dạng sóng
    end_time = torch.cumsum(phone_duration, dim=-1)
    start_time = end_time - phone_duration
    start_frame = start_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    end_frame = end_time / 1000 * hps.data.sampling_rate / hps.data.hop_length
    spec_length = end_frame.max(dim=-1).values
    pos = torch.arange(0, spec_length.item(), device=device)
    attn = torch.logical_and(
        pos[None, :, None] >= start_frame[:, None, :],
        pos[None, :, None] < end_frame[:, None, :],
    ).float()
    with torch.inference_mode():
        y_hat = generator.infer(
            phone_idx, phone_length, spec_length, attn, max_len=None, noise_scale=0.667
        )[0]
    wave = y_hat[0, 0].data.cpu().numpy()
    return (wave * (2**15)).astype(np.int16)

# Load mô hình 
@st.cache(allow_output_mutation=True)
def load_models():
    duration_net = DurationNet(hps.data.vocab_size, 64, 4).to(device)
    duration_net.load_state_dict(torch.load(duration_model_path, map_location=device))
    duration_net = duration_net.eval()
    generator = SynthesizerTrn(
        hps.data.vocab_size,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **vars(hps.model),
    ).to(device)
    del generator.enc_q
    ckpt = torch.load(lightspeed_model_path, map_location=device)
    params = {}
    for k, v in ckpt["net_g"].items():
        k = k[7:] if k.startswith("module.") else k
        params[k] = v
    generator.load_state_dict(params, strict=False)
    del ckpt, params
    generator = generator.eval()
    return duration_net, generator


# Đường dẫn đến thư mục chứa file txt với nội dung truyện ngắn
stories_folder = "stories"

# Lấy danh sách các file txt trong thư mục
story_files = [file for file in os.listdir(stories_folder) if file.endswith(".txt")]

# Tạo một từ điển để lưu trữ nội dung của từng truyện ngắn
story_contents = {}
for file in story_files:
    with open(os.path.join(stories_folder, file), "r", encoding="utf-8") as f:
        story_contents[file[:-4]] = f.read()
        

# Streamlit app
st.title("Audio Book")

# Hiển thị danh sách các truyện ngắn để chọn
selected_story = st.selectbox("Chọn một truyện ngắn", list(story_contents.keys()))

# Hiển thị nội dung của truyện ngắn
st.markdown(f"## {selected_story}")
st.text(story_contents[selected_story])

# Process text and generate speech
if st.button("Tạo Giọng Đọc"):
    # Lấy đoạn trích được chọn
    selected_text = story_contents[selected_story]

    # Xử lý văn bản và tạo giọng nói (sử dụng hàm text_to_speech)
    duration_net, generator = load_models()
    paragraphs = selected_text.split("\n")
    clips = []  # list of audio clips
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph == "":
            continue
        clips.append(text_to_speech(duration_net, generator, paragraph))
    audio = np.concatenate(clips)

    # Hiển thị audio
    sample_rate = hps.data.sampling_rate  # Thay bằng giá trị thực tế của tần số lấy mẫu
    st.audio(audio, format="audio/wav", start_time=0, sample_rate=sample_rate)
