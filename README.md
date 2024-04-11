"streamlit run .\app.py" trên Terminal để chạy mô hình TTS

config/: chứa các thông số cấu hình của mô hình, như kích thước mô hình
duration_model/: Mô hình này thường dự đoán thời lượng của các đơn vị (như âm tiết) trong văn bản.
model_Tacotron/: đến mô hình Tacotron tổng hợp giọng nói từ văn bản.
vbx_phone_set.json/: ánh xạ giữa các đơn vị âm tiết và các nhóm nguyên âm, phụ âm