# Mental Health Analysis - Data Science Final Project

**Môn học:** CSC17104 - Lập trình cho Khoa học Dữ liệu  
**Trường:** Đại học Khoa học Tự nhiên, ĐHQG-HCM  
**Khoa:** Công nghệ Thông tin

---

## 📋 Tổng Quan Dự Án

Dự án này phân tích các mô hình sức khỏe tinh thần tại nơi làm việc sử dụng các kỹ thuật khoa học dữ liệu. Chúng tôi khám phá các yếu tố ảnh hưởng đến sức khỏe tinh thần, hành vi tìm kiếm điều trị và trải nghiệm tại nơi làm việc thông qua phân tích dữ liệu toàn diện và machine learning.

### Thông Tin Nhóm

**Nhóm 13:**

| MSSV     | Họ và Tên        |
| -------- | ---------------- |
| 23122021 | Bùi Duy Bảo      |
| 23122032 | Nguyễn Việt Hùng |
| 23122039 | Huỳnh Trung Kiệt |

### Phân Chia Công Việc

![Task Allocation](/figure/phancong.png)

_Figure 1. Bảng phân chia công việc giữa các thành viên._

🔗 **Google Sheets (full resolution):**  
https://docs.google.com/spreadsheets/d/1cQzm5i8q_oGommrFgbFqNS2OQBbHz8ctOng-KI3iLe0/edit?usp=sharing

## 📊 Dataset - Bộ Dữ Liệu

**Nguồn:** Mental Health Dataset  
**Nền tảng:** Kaggle / OSMI (Open Sourcing Mental Illness)  
**URL:** https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset

**Mô tả:**  
Dữ liệu khảo sát về trải nghiệm sức khỏe tinh thần tại nơi làm việc, chủ yếu từ ngành công nghệ. Bộ dữ liệu chứa phản hồi từ ~292,000 cá nhân trên nhiều quốc gia, được thu thập từ năm 2014-2016.

**Kích thước:**

- **Dòng:** 292,364 phản hồi khảo sát
- **Cột:** 17 đặc trưng
- **Định dạng:** CSV

**Đặc trưng chính:**

- **Demographics (Nhân khẩu học):** Gender, Country, Occupation
- **Mental Health History:** Family history, personal history, treatment seeking
- **Workplace Factors:** Days indoors, work interest, social weakness
- **Attitudes:** Care options awareness, mental health interview
- **Symptoms:** Mood swings, growing stress, coping struggles, habit changes

---

## 🎯 Research Questions - Câu Hỏi Nghiên Cứu

Chúng tôi đã xây dựng **6 câu hỏi nghiên cứu có ý nghĩa** để định hướng phân tích:

### Câu hỏi 1: Các yếu tố nào dự đoán mạnh nhất việc tìm kiếm điều trị sức khỏe tinh thần? (Machine Learning)

Sử dụng phân loại machine learning để xác định các yếu tố dự đoán chính của hành vi tìm kiếm điều trị.

**Phương pháp:** Classification với Random Forest, Logistic Regression, Decision Tree, Gradient Boosting

### Câu hỏi 2: Lịch sử gia đình ảnh hưởng như thế nào đến sức khỏe tinh thần và trải nghiệm tại nơi làm việc?

Phân tích thống kê tác động của family history lên tình trạng sức khỏe tinh thần hiện tại và các yếu tố công việc.

**Phương pháp:** Statistical Analysis, Chi-square test, Correlation Analysis

### Câu hỏi 3: Có sự khác biệt về địa lý/văn hóa trong thái độ về sức khỏe tinh thần không?

Phân tích so sánh thái độ về sức khỏe tinh thần và các mô hình điều trị giữa các quốc gia.

**Phương pháp:** Comparative Analysis, Visualization

### Câu hỏi 4: Các triệu chứng nào thường xuất hiện cùng nhau ở những người gặp khó khăn trong đối phó?

Phân tích co-occurrence patterns của các triệu chứng.

**Phương pháp:** Pattern Analysis, Correlation Heatmap

### Câu hỏi 5: Các yếu tố liên quan đến công việc có xu hướng kết hợp với nhau như thế nào?

Phân tích phân cụm để xác định các nhóm risk profile khác nhau.

**Phương pháp:** K-Means Clustering

### Câu hỏi 6: Thái độ về sức khỏe tinh thần tại nơi làm việc có thay đổi theo thời gian không?

Phân tích xu hướng theo thời gian của treatment seeking và awareness.

**Phương pháp:** Time Series Analysis, Trend Analysis

---

## 🔍 Key Findings - Phát Hiện Chính

### Câu hỏi 1: Machine Learning - Dự đoán tìm kiếm điều trị

- **Mô hình tốt nhất:** Decision Tree đạt F1-Score cao nhất (0.7856, Accuracy: 0.7715)
- **Top 5 yếu tố quan trọng nhất:**
  1. Mental Health History (lịch sử sức khỏe tinh thần cá nhân)
  2. Family History (tiền sử gia đình)
  3. Mood Swings (thay đổi tâm trạng)
  4. Coping Struggles (khó khăn đối phó)
  5. Growing Stress (căng thẳng gia tăng)
- **Insight:** Các yếu tố lịch sử và triệu chứng quan trọng hơn các yếu tố môi trường làm việc

### Câu hỏi 2: Ảnh hưởng của lịch sử gia đình

- **Người có family history:** 73.17% tìm kiếm điều trị
- **Người không có family history:** 35.66% tìm kiếm điều trị (chênh lệch 37.51%)
- Family history tương quan mạnh với Mental Health History, Mood Swings, Growing Stress (p < 0.001)
- **Insight:** Yếu tố di truyền ảnh hưởng đến cả triệu chứng lẫn hành vi tìm kiếm điều trị

### Câu hỏi 3: Khác biệt địa lý/văn hóa

- **Quốc gia có tỷ lệ cao nhất:** Australia (60.44% tìm kiếm điều trị)
- **Quốc gia có tỷ lệ thấp nhất:** France (0.00% tìm kiếm điều trị)
- **Chênh lệch:** Lên đến 60.44% giữa các quốc gia
- **Top countries:** US, UK, Canada, Australia có awareness và treatment rate cao nhất
- **Insight:** Văn hóa và stigma đóng vai trò quan trọng trong thái độ về mental health

### Câu hỏi 4: Patterns triệu chứng

- **Co-occurrence cao:**
  - Mental Health History + Mood Swings (>70% overlap)
  - Growing Stress + Changes in Habits
  - Coping Struggles + Social Weakness
- Trung bình **1.36 triệu chứng** đồng thời ở nhóm có Coping Struggles
- **Insight:** Coping Struggles là chỉ số tổng hợp tốt cho mức độ nghiêm trọng

### Câu hỏi 5: Patterns yếu tố công việc

- **3 Clusters được xác định:**
  1. **Cluster 0 (23.5%):** High Risk - Stress cao, mất hứng thú, yếu kém xã hội
  2. **Cluster 1 (47.3%):** Moderate Risk/Healthy - Có một số vấn đề nhưng chưa nghiêm trọng
  3. **Cluster 2 (29.2%):** Mixed - Kết hợp các đặc điểm khác nhau
- Growing Stress ↔ Work Interest có correlation mạnh (r > 0.5)

### Câu hỏi 6: Xu hướng theo thời gian

- **Xu hướng ổn định:** Không có thay đổi rõ rệt (slope ≈ 0)
- Tỷ lệ treatment seeking và awareness dao động nhưng không có trend tăng/giảm
- **Insight:** Cần các can thiệp mạnh mẽ hơn để thay đổi thái độ về mental health

---

## 🗂️ File Structure - Cấu Trúc File

```
Final_project/
│
├── Mental_Health_Dataset.csv          # Dữ liệu gốc
├── main.ipynb                         # Notebook phân tích chính
├── README.md                          # File này
└── requirements.txt                   # Các thư viện Python cần thiết
```

---

## 🚀 Hướng Dẫn Chạy - How to Run

### Yêu Cầu - Prerequisites

**Phiên bản Python yêu cầu:** Python 3.8 trở lên

**Thư viện yêu cầu - Required Libraries:**

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
jupyter
warnings
```

### Các Bước Cài Đặt - Installation Steps

1. **Clone hoặc tải xuống repository này**

2. **Cài đặt các thư viện phụ thuộc:**

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
```

Hoặc nếu có file `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. **Khởi chạy Jupyter Notebook:**

```bash
jupyter notebook
```

4. **Mở file `main.ipynb`**

5. **Chạy tất cả các cell:**
   - Cách 1: Click "Cell" → "Run All"
   - Cách 2: Sử dụng Shift+Enter để chạy từng cell
   - Cách 3: Click "Kernel" → "Restart & Run All" để chạy mới hoàn toàn

### Thời Gian Chạy Dự Kiến - Expected Runtime

- Chạy toàn bộ notebook: ~5-15 phút (tùy thuộc vào phần cứng)
- Huấn luyện machine learning: ~2-5 phút
- Khám phá dữ liệu: ~1-2 phút

---

## 📦 Dependencies - Các Thư Viện Phụ Thuộc

### Core Libraries - Thư viện Cốt lõi

- **pandas** (>=1.3.0): Thao tác và phân tích dữ liệu
- **numpy** (>=1.21.0): Tính toán số học
- **matplotlib** (>=3.4.0): Trực quan hóa dữ liệu
- **seaborn** (>=0.11.0): Trực quan hóa dữ liệu thống kê
- **scipy** (>=1.7.0): Tính toán khoa học và thống kê

### Machine Learning

- **scikit-learn** (>=0.24.0): Các thuật toán và công cụ machine learning

### Notebook Environment

- **jupyter** (>=1.0.0): Môi trường notebook tương tác

---

## 📝 Methodology - Phương Pháp Luận

### 1. Data Collection - Thu thập Dữ liệu

- Ghi chép nguồn dữ liệu và ngữ cảnh
- Xác định giấy phép và quyền sử dụng
- Giải thích phương pháp thu thập và các hạn chế

### 2. Data Exploration - Khám phá Dữ liệu

- Phân tích cấu trúc và chất lượng bộ dữ liệu
- Kiểm tra phân phối của tất cả các biến
- Xác định mô hình dữ liệu thiếu
- Tính toán tương quan và mối quan hệ
- Trực quan hóa các mô hình và xu hướng chính

### 3. Question Formulation - Xây dựng Câu hỏi

- Phát triển 6 câu hỏi nghiên cứu có ý nghĩa và thách thức
- Đảm bảo các câu hỏi cung cấp giá trị thực tiễn
- Bao gồm ít nhất 1 câu hỏi machine learning

### 4. Data Analysis - Phân tích Dữ liệu

- **Preprocessing:** Làm sạch dữ liệu, xử lý giá trị thiếu, mã hóa biến
- **Analysis:** Áp dụng các kỹ thuật thống kê và ML phù hợp
- **Visualization:** Tạo các biểu đồ rõ ràng, có thông tin
- **Interpretation:** Rút ra các kết luận có ý nghĩa

### 5. Conclusions - Kết luận

- Tóm tắt các phát hiện chính
- Thừa nhận các hạn chế
- Đề xuất hướng phát triển trong tương lai

---

## 💡 Technical Highlights - Điểm Nổi Bật Kỹ Thuật

### Machine Learning Approach

- Huấn luyện nhiều mô hình phân loại: **Logistic Regression, Decision Tree, Random Forest, Gradient Boosting**
- Thực hiện feature engineering và encoding
- Đánh giá sử dụng nhiều metrics: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**
- Diễn giải feature importance để hiểu yếu tố quan trọng

### Statistical Analysis

- **Chi-square tests** cho các liên kết phân loại
- **Correlation analysis** để khám phá mối quan hệ
- Thống kê so sánh giữa các nhóm
- Time series trend analysis

### Data Visualization

- Distribution plots (histograms, bar charts, pie charts)
- Correlation heatmaps
- Confusion matrices
- Feature importance visualizations
- Temporal trend plots
- Geographic comparison charts

---

## ⚠️ Limitations - Hạn Chế

### Hạn chế về Dataset

- **Selection Bias:** Dữ liệu từ khảo sát tự nguyện, có thể thiên về nhóm quan tâm đến mental health
- **Geographical Bias:** Phần lớn dữ liệu từ US và các nước phương Tây
- **Gender Imbalance:** ~82% nam giới, có thể tạo bias trong mô hình
- **Self-reported Data:** Dữ liệu tự báo cáo có thể không chính xác hoàn toàn
- **Industry Specific:** Chủ yếu từ ngành công nghệ, khó khái quát cho các ngành khác

### Hạn chế về Phân tích

- **Correlation ≠ Causation:** Phân tích tương quan không chứng minh nhân quả
- **Cross-sectional:** Dữ liệu tại một thời điểm, không theo dõi dài hạn
- **K-means Limitations:** Clustering có thể đơn giản hóa quá mức các patterns phức tạp
- **Missing Context:** Không có thông tin chi tiết về môi trường làm việc cụ thể

---

## 🔮 Future Work - Hướng Phát triển Tương lai

### Câu hỏi Nghiên cứu Bổ sung

- Các yếu tố nguy cơ tương tác với nhau như thế nào?
- Có sự khác biệt về giới tính trong trải nghiệm mental health không?
- Ảnh hưởng của self-employment đến sức khỏe tinh thần?
- Mối quan hệ giữa treatment và outcomes (nếu có dữ liệu)?

### Phương pháp Nâng cao

- **Deep Learning** approaches cho dự đoán
- **Causal Inference** techniques để xác định nhân quả
- **Network Analysis** để hiểu mối quan hệ giữa các triệu chứng
- **Interactive Dashboards** để trực quan hóa

### Dữ liệu Bổ sung Cần thiết

- Đo lường mức độ nghiêm trọng của triệu chứng (severity measures)
- Kết quả điều trị và follow-up
- Chính sách workplace về mental health
- Yếu tố kinh tế-xã hội chi tiết
- **Longitudinal data** để theo dõi thay đổi theo thời gian

---

## 📚 References - Tài Liệu Tham Khảo

1. Open Sourcing Mental Illness (OSMI) - https://osmihelp.org/
2. Kaggle Mental Health Datasets - https://www.kaggle.com/datasets/bhavikjikadara/mental-health-dataset
3. WHO Mental Health in the Workplace - https://www.who.int/mental_health/in_the_workplace/en/
4. Scikit-learn Documentation - https://scikit-learn.org/
5. Pandas Documentation - https://pandas.pydata.org/

---

## 📧 Contact - Liên Hệ

Để có câu hỏi hoặc phản hồi về dự án này:

- **Nhóm:** Nhóm 13
- **Môn học:** CSC17104 - Lập Trình Cho Khoa Học Dữ Liệu
- **Trường:** Đại học Khoa học Tự nhiên, ĐHQG-HCM

**Thành viên:**

- Bùi Duy Bảo - 23122021
- Nguyễn Việt Hùng - 23122032
- Huỳnh Trung Kiệt - 23122039

---

## 📄 License - Giấy Phép

Dự án này được nộp như bài tập học thuật cho môn CSC17104 - Lập Trình Cho Khoa Học Dữ Liệu tại Đại học Khoa học Tự nhiên, ĐHQG-HCM.

Bộ dữ liệu tuân theo giấy phép **CC0: Public Domain** từ nguồn dữ liệu (OSMI/Kaggle). Phân tích này chỉ dành cho mục đích giáo dục.

---

**Cập nhật lần cuối:** Tháng 12 năm 2025  
**Phiên bản:** 1.0  
**Trạng thái:** Hoàn thành
