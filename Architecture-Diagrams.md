# Epi-SynLethal: Sequence Diagrams

Tài liệu này mô tả các luồng hoạt động chính của hệ thống thông qua các sơ đồ tuần tự (Sequence Diagrams) sử dụng Mermaid.

---

## 1. Luồng Xác thực (Split-Token Authentication Flow)

Mô tả cách hệ thống đăng nhập, lưu trữ bảo mật bằng HttpOnly Cookie và tự động cấp mới (auto-refresh) Access Token khi hết hạn mà không làm gián đoạn trải nghiệm người dùng.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Frontend as Frontend (React)
    participant AuthAPI as Auth API (/api/auth)
    participant DB as Database (PostgreSQL)

    User->>Frontend: Nhập Username & Password
    Frontend->>AuthAPI: POST /login (username, password)
    AuthAPI->>DB: Xác thực credentials
    DB-->>AuthAPI: OK (User Data)
    AuthAPI->>AuthAPI: Generate Access Token (15m)
    AuthAPI->>AuthAPI: Generate Refresh Token (7d)
    AuthAPI-->>Frontend: Trả về Access Token (JSON) & Set-Cookie (HttpOnly Refresh Token)

    note over Frontend: Lưu Access Token vào LocalStorage

    User->>Frontend: Chuyển trang / Yêu cầu dữ liệu
    Frontend->>AuthAPI: Kèm Access Token trong Header (Bearer)

    alt Access Token Hết Hạn
        AuthAPI-->>Frontend: 401 Unauthorized
        Frontend->>AuthAPI: POST /refresh (gửi ngầm HttpOnly Cookie)
        AuthAPI->>AuthAPI: Verify Refresh Token
        AuthAPI-->>Frontend: Trả về Access Token mới
        Frontend->>AuthAPI: Retry request ban đầu với Token mới
        AuthAPI-->>Frontend: 200 OK (Data)
    else Access Token Hợp Lệ
        AuthAPI-->>Frontend: 200 OK (Data)
    end
```

---

## 2. Luồng Theo dõi Hành vi (Behavior Tracking Flow)

Mô tả cách Frontend thu thập hành vi người dùng (nhìn ngắm sản phẩm, thêm vào giỏ, tìm kiếm) và gửi về Backend theo từng lô (batch) để tối ưu hiệu suất mạng.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as UI Components
    participant Tracker as Tracker Service (tracker.ts)
    participant API as Behavior API (/api/behavior/track)
    participant DB as Database (PostgreSQL)

    User->>UI: Xem sản phẩm A (> 3 giây)
    UI->>Tracker: trackView(A, duration)
    note over Tracker: Push event vào Buffer (Memory)

    User->>UI: Tìm kiếm "Key Ring"
    UI->>Tracker: trackSearch("Key Ring")
    note over Tracker: Push event vào Buffer

    loop Mỗi 10 giây (hoặc khi chuyển trang/thêm giỏ hàng)
        Tracker->>API: POST /track (Batch Array of Events)
        API->>DB: Bulk insert vào bảng behavior_events
        DB-->>API: Success
        API-->>Tracker: 200 OK
        Tracker->>Tracker: Xóa Buffer
        Tracker->>UI: Invalidate Recommendation Cache (kích hoạt load lại gợi ý)
    end
```

---

## 3. Luồng Voucher AI Động (Dynamic AI Voucher Flow)

Mô tả luồng người dùng vào Giỏ hàng (Cart) và hệ thống Machine Learning ngầm tính toán xác suất mua hàng để tung ra Voucher cá nhân hóa.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Cart as Cart UI (OrderSummary)
    participant PromoAPI as Promotions API
    participant ML as Behavior Engine (Random Forest)
    participant DB as Database

    User->>Cart: Truy cập Giỏ hàng (chọn các sản phẩm)
    note over Cart: Debounce 1s chờ user dừng thao tác
    Cart->>PromoAPI: GET /dynamic-voucher?cart_total=X

    PromoAPI->>DB: Lấy lịch sử hành vi (View, Cart, Cancel rate)
    DB-->>PromoAPI: Raw Data

    PromoAPI->>ML: Trích xuất RFM Features
    ML->>ML: Chạy Random Forest Model
    ML-->>PromoAPI: Trả về Purchase Probability (VD: 45%)

    alt Xác suất 30% - 50% (Cần mồi câu mạnh)
        PromoAPI-->>Cart: Voucher -20% (Kèm lý do: Nudge)
    else Xác suất 50% - 70% (Đang phân vân)
        PromoAPI-->>Cart: Voucher -10%
    else Xác suất 70% - 90% (Sắp mua)
        PromoAPI-->>Cart: Voucher -5% (hoặc Freeship)
    else Xác suất > 90% hoặc < 30%
        PromoAPI-->>Cart: Null (Không cấp Voucher để tối ưu lợi nhuận)
    end

    Cart-->>User: Hiển thị Banner Voucher AI động
```

---

## 4. Luồng Gợi ý Sản phẩm (Product Recommendation Flow)

Mô tả cách thuật toán KNN kết hợp với xử lý ngôn ngữ tự nhiên (TF-IDF) tìm ra các sản phẩm liên quan để cross-sell.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Homepage / Product Detail
    participant RecAPI as Recommendation API
    participant Engine as Behavior Engine (KNN + TF-IDF)
    participant DB as Database

    User->>UI: Cuộn xuống mục "Có thể bạn sẽ thích"
    UI->>RecAPI: GET /recommendations?source_product_id=X (Tùy chọn)

    RecAPI->>DB: Lấy Lịch sử Hành vi gần nhất (View, Cart, Search) của User
    DB-->>RecAPI: Danh sách source products

    RecAPI->>Engine: Gửi danh sách nguồn vào KNN Pipeline
    note over Engine: Load Ma trận TF-IDF (Tên, Mô tả, Category)
    note over Engine: Load Product Scaler (Giá cả, Lượt mua)

    Engine->>Engine: Tìm K láng giềng gần nhất (Cosine Similarity)
    Engine-->>RecAPI: Danh sách Product IDs (Đã được chấm điểm Tương đồng)

    RecAPI->>DB: Fetch chi tiết các Product IDs đó
    DB-->>RecAPI: Product Data

    RecAPI-->>UI: JSON (Top N Recommendations)
    UI-->>User: Hiển thị Slider/Grid sản phẩm gợi ý
```

---

## 5. Luồng Cập nhật & Phân cụm ngầm (Customer Segmentation Cron/Background)

Luồng phân cụm dữ liệu Admin Dashboard (K-Means). Thực tế luồng này có thể chạy ngầm (cron job) hoặc khi Admin truy cập Dashboard.

```mermaid
sequenceDiagram
    autonumber
    actor Admin
    participant Dashboard as Admin Dashboard
    participant API as Admin API
    participant ML as ML Engine (K-Means & PCA)
    participant DB as Database

    Admin->>Dashboard: Mở trang chủ Admin
    Dashboard->>API: Lấy dữ liệu Segmentation

    API->>DB: Lấy dữ liệu RFM của toàn bộ Users
    DB-->>API: Raw Data

    API->>ML: Tiền xử lý (StandardScaler)
    ML->>ML: Giảm chiều dữ liệu (PCA)
    ML->>ML: Chạy thuật toán K-Means Clustering
    ML-->>API: Trả về Clusters (VIP, Tiềm năng, Vãng lai, Ngủ đông) & Silhouette Score

    API-->>Dashboard: Dữ liệu phân cụm (Counts + Metrics)
    Dashboard-->>Admin: Hiển thị biểu đồ Bar Chart & Thống kê Model
```
