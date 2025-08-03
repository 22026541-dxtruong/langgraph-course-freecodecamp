# AI Agent with LangGraph and Gemini

### Yêu cầu

* **Python 3.9+**
* **Khóa API Gemini**
* `pip` để cài đặt các thư viện

### Cài đặt

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/22026541-dxtruong/langgraph-course-freecodecamp](https://github.com/22026541-dxtruong/langgraph-course-freecodecamp)
    cd langgraph-course-freecodecamp
    ```

2.  **Tạo môi trường ảo:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Trên macOS/Linux
    # venv\Scripts\activate  # Trên Windows
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Thiết lập khóa API Gemini:**
    Tạo một file `.env` trong thư mục gốc của dự án và thêm khóa API vào đó (lấy khóa này từ [Google AI Studio](https://aistudio.google.com/)):
    ```
    GOOGLE_API_KEY="gemini-api-key"
    ```