# Epi-SynLethal (E-commerce & ML Recommendation System)
> Một nền tảng thương mại điện tử toàn diện kết hợp Machine Learning để phân tích hành vi người dùng, phân cụm khách hàng, và gợi ý sản phẩm/khuyến mãi cá nhân hóa theo thời gian thực.

---

## 1. Tổng quan Dự án (Project Overview)

Dự án này là một hệ thống e-commerce hoàn chỉnh từ Frontend đến Backend, tích hợp sâu một pipeline Machine Learning (ML). Khác với các hệ thống e-commerce truyền thống chỉ có các quy tắc cứng (hardcoded rules), hệ thống này sử dụng AI để tự động hóa việc đưa ra quyết định marketing (như cấp phát voucher) dựa trên xác suất mua hàng của từng người dùng cụ thể.

### Tech Stack
- **Frontend:** React, Vite, TypeScript, TanStack Router, Zustand, Tailwind CSS, Shadcn UI.
- **Backend:** Python, FastAPI, SQLAlchemy, PostgreSQL (AsyncPG), Scikit-Learn, Pandas.
- **Machine Learning:** Random Forest (dự đoán xác suất mua), K-Means + PCA (phân cụm), KNN (gợi ý sản phẩm).

---

## 2. Kiến trúc Hệ thống (Architecture)

### 2.1. Split-Token Authentication (Bảo mật)
- **Access Token (JWT):** Lưu trong `localStorage` trên Frontend, được đính kèm vào Header mỗi request. Có thời hạn ngắn (15-30 phút).
- **Refresh Token:** Lưu dưới dạng `HttpOnly, Secure Cookie`. Giúp hệ thống chống lại các cuộc tấn công XSS (kẻ cắp không thể đọc được cookie) và CSRF (token tự động gửi ngầm).
- Frontend sử dụng Axios Interceptors để tự động (auto) gọi API refresh token mỗi khi Access Token hết hạn, mang lại trải nghiệm mượt mà không đứt đoạn cho User.

### 2.2. Behavior Tracking Engine (Theo dõi hành vi)
- Hệ thống bắt các sự kiện (events) của người dùng như `view` (xem sản phẩm với độ trễ > 3 giây), `add_to_cart`, `remove_from_cart`, `search` và gửi về Backend thông qua API `/api/behavior/track` dưới dạng batch (gom nhóm mỗi 10 giây để giảm tải).
- Dữ liệu này được lưu vào bảng `behavior_events` và là đầu vào (input) quan trọng cho toàn bộ ML Pipeline.

---

## 3. Các tính năng cốt lõi đã hoàn thiện (Implemented Features)

### 3.1. Dành cho Người dùng (Client-Facing)
- **Cửa hàng & Giỏ hàng:** Liệt kê sản phẩm, phân trang, gom nhóm các **biến thể (Variants)** thành một sản phẩm đại diện trên trang chủ để UI gọn gàng. Hỗ trợ chọn biến thể (VD: Các chữ cái của móc khóa) trong trang Chi tiết.
- **Search:** Tìm kiếm sản phẩm theo tên/mô tả. Lịch sử tìm kiếm được tracking để làm dữ liệu KNN.
- **Checkout & Khuyến mãi:** Áp dụng mã giảm giá. Tính toán tự động tổng tiền. **Đặc biệt:** Hệ thống có banner hiển thị Voucher AI linh động dựa vào độ "ngập ngừng" của khách khi mua sắm.
- **Recommendations:** Trang chủ và trang chi tiết sản phẩm hiển thị gợi ý (Cross-sell & Up-sell) được sinh ra từ thuật toán KNN + TF-IDF (Text matching).

### 3.2. Dành cho Quản trị viên (Admin-Facing)
- **Dashboard Phân tích:** Hiển thị số liệu Real-time về sức khỏe của AI Models (CV F1 Score, Silhouette Score). Đặc biệt, hệ thống vẽ biểu đồ thanh ngang (bar chart) phân cụm người dùng (Customer Segmentation) từ dữ liệu K-Means.
- **Quản lý Đơn hàng (Orders):** Danh sách đơn hàng phân trang, xem chi tiết order, theo dõi doanh thu.
- **Quản lý Sản phẩm (Products CRUD):** Thêm mới, chỉnh sửa, xóa và tìm kiếm sản phẩm.
- **Promotions:** Trình quản lý mã giảm giá (Percentage/Fixed) theo từng target audience cụ thể.

---

## 4. Pipeline Machine Learning (AI Models)

Đây là "trái tim" của dự án, biến một trang web bán hàng bình thường thành một hệ thống AI-driven.

### 4.1. Phân cụm khách hàng (Customer Segmentation)
- **Thuật toán:** PCA (để giảm chiều dữ liệu từ hàng chục features xuống còn 2-3) + **K-Means Clustering**.
- **Cách thức:** Chạy ngầm dựa trên mô hình RFM (Recency - thời gian mua gần nhất, Frequency - tần suất mua, Monetary - số tiền tiêu thụ).
- **Kết quả:** Phân khách hàng thành 4 cụm chính:
  - 🏆 *VIP / Champions*
  - 🌱 *Potential / Tiềm năng*
  - 🚶 *Casual / Vãng lai*
  - 💤 *Hibernating / Ngủ đông*
- Dữ liệu này được báo cáo trực tiếp lên **Admin Dashboard**.

### 4.2. Khuyến mãi thông minh (Dynamic AI Voucher)
- **Thuật toán:** **Random Forest Classifier**.
- **Cách thức:** Model phân tích hành vi của user (số lần xem, thêm vào giỏ, thời gian xem, tỷ lệ hủy đơn) để tính ra `Purchase Probability` (xác suất thanh toán giỏ hàng).
- **Ứng dụng:**
  - Nếu xác suất `< 30%`: Bỏ qua (User chỉ xem dạo, tặng voucher cũng lãng phí).
  - Xác suất `30% - 50%`: Khách cần mồi câu mạnh → Tặng voucher giảm 20%.
  - Xác suất `50% - 70%`: Khách đang phân vân → Tặng voucher giảm 10%.
  - Xác suất `70% - 90%`: Khách sắp mua → Tặng Freeship hoặc giảm 5% để đẩy nhanh tiến độ.
  - Xác suất `> 90%`: Khách chắc chắn mua → Không tặng gì để tối ưu hóa biên lợi nhuận.

### 4.3. Gợi ý Sản phẩm (Product Recommendations)
- **Thuật toán:** **KNN (K-Nearest Neighbors)** kết hợp với Ma trận TF-IDF.
- **Nguồn dữ liệu:** Dựa trên lịch sử Click (Views), Tìm kiếm (Search), và Giỏ hàng (Cart).
- **Ứng dụng:** Tìm ra các sản phẩm có thuộc tính text hoặc được mua chung với nhau nhiều nhất để hiển thị ở mục "Có thể bạn sẽ thích".

---

## 5. Những cải tiến cuối cùng (Tech Debt Solved)
1. **Gộp nhóm biến thể (Product Grouping):** Xóa bỏ tình trạng rác trang chủ do hàng trăm mẫu mã giống hệt nhau (chỉ khác màu/chữ cái). Hệ thống dùng `parent_sku` để gom nhóm thông minh cả ở DB và API.
2. **Cancellation Rate:** Bỏ cơ chế hardcode, tỷ lệ hủy đơn (`CANCELED`) giờ được truy vấn thực tế bằng SQL count trên bảng `Orders` → giúp Random Forest dự đoán chính xác hơn những tài khoản có hành vi "bùng hàng".
3. **Stock Control:** Khi checkout, hệ thống tự động tăng `purchase_count` và `num_customers`.

---

## 6. Kế hoạch trong Tương lai (Future Expansion)
Dù dự án đã rất hoàn thiện ở mức độ Đồ án/MVP, nếu triển khai lên Production thực tế, hệ thống có thể mở rộng thêm:
- **Redis Caching:** Cache lại các API tính toán gợi ý (Recommendations) và RFM để giảm thiểu số lượng JOIN/GROUP BY lên PostgreSQL.
- **Thanh toán thực:** Thay thế luồng Mock Checkout hiện tại bằng webhook của Stripe, PayPal hoặc VNPay.
- **Real-time Pipeline (Kafka/RabbitMQ):** Tách Behavior Engine thành một microservice độc lập dùng Message Queue xử lý hàng triệu event click stream mà không làm chậm server chính.
- **Unit & Integration Tests:** Phủ thêm test cases (Pytest cho Backend và Vitest cho Frontend) để bảo vệ logic ML không bị phá vỡ khi thay đổi code.
