# Tool: Generate Test Cases from Spec using Gemini API + Streamlit UI
# Requirement: pip install google-generativeai pandas streamlit

import google.generativeai as genai
import pandas as pd
import streamlit as st
import io
import re  # Thêm import re để sử dụng regex

# ====== CONFIGURATION ======
#GEMINI_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBkE5L64hDGtumVvFz8ryTmlq8LjDjhJ9A"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# ====== FUNCTION TO PARSE MARKDOWN TABLE TO PANDAS DF ======
def parse_test_cases_to_df(markdown_table):
    try:
        table_io = io.StringIO(markdown_table)
        lines = [line.strip() for line in table_io if line.strip().startswith("|")]
        cleaned_table = "\n".join(lines)
        df = pd.read_csv(io.StringIO(cleaned_table), sep="|", engine="python")
        df = df.dropna(axis=1, how="all")
        df.columns = [col.strip() for col in df.columns]
        df = df[1:].reset_index(drop=True)

        return df
    except Exception as e:
        st.error("Failed to parse table. Showing raw output.")
        return None

# ====== FUNCTION TO GENERATE TEST VIEWPOINTS ======
def generate_test_viewpoints(spec_text, language):
    prompt = f"""
    You are a QA test analyst.
    Given the following system specification, generate a list of test viewpoints.
    Test viewpoints are high-level testing perspectives or scenarios derived from the specification.
    Provide the viewpoints in a numbered list format.
    Just generate the test viewpoint without any additional text.

    Language: {language}

    Specification:
    {spec_text}
    """
    response = model.generate_content(prompt)
    content = response.text
    return content

# ====== FUNCTION TO GENERATE TEST CASES BY VIEWPOINT ======
def generate_test_cases_by_viewpoint(viewpoints_text, language):
    """
    Generate test cases for each viewpoint and group them.
    """
    grouped_test_cases = {}
    viewpoints = [vp.strip() for vp in viewpoints_text.split("\n") if vp.strip()]

    for i, viewpoint in enumerate(viewpoints, start=1):
        prompt = f"""
        You are a QA test analyst.
        Given the following test viewpoint, generate a test case table.
        Just generate the test case table without any additional text.
        Use this format:
        | Test Case ID | Description | Step | Expected Output | Priority |

        Language: {language}

        Test Viewpoint:
        {viewpoint}
        """
        response = model.generate_content(prompt)
        content = response.text
        grouped_test_cases[f"Viewpoint {viewpoint}"] = content

    return grouped_test_cases

# ====== STREAMLIT UI ======
st.title("🧪 Test Case Generator from Spec AI Agent")

# Nhập đặc tả hệ thống
spec_input = st.text_area("Enter system specification:", height=200, key="spec_input")

# Biến lưu trữ Test Viewpoints
viewpoints_text = st.session_state.get("viewpoints_text", "")

# Thêm lựa chọn ngôn ngữ
language = st.selectbox(
    "Select language for Test Viewpoints and Test Cases:",
    ["English", "Vietnamese"],  # Thêm các ngôn ngữ bạn muốn hỗ trợ
    key="language_select"
)

# Nút Generate Test Viewpoints
if st.button("Generate Test Viewpoints"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.error("❗ Please update your Gemini API key in the script.")
    elif not spec_input.strip():
        st.warning("Please enter a system specification.")
    else:
        with st.spinner("Generating test viewpoints, please wait..."):
            viewpoints_text = generate_test_viewpoints(spec_input, language)
            st.session_state["viewpoints_text"] = viewpoints_text  # Lưu vào session state
            st.success("✅ Test viewpoints generated successfully!")
            st.text_area("Test Viewpoints:", value=viewpoints_text, height=300, key="viewpoints_output")

# Nút Generate Test Cases từ Test Viewpoints
if viewpoints_text and st.button("Generate Test Cases from Viewpoints"):
    with st.spinner("Generating test cases, please wait..."):
        viewpoints = [vp.strip() for vp in viewpoints_text.split("\n") if vp.strip()]
        progress_bar = st.progress(0)  # Khởi tạo thanh tiến trình
        status_text = st.empty()  # Tạo vùng trống để hiển thị trạng thái
        grouped_test_cases = generate_test_cases_by_viewpoint(viewpoints_text, language)

        # Tạo danh sách để lưu tất cả các Test Cases
        all_test_cases = []

        # Hiển thị Test Cases theo từng Viewpoint
        for i, (viewpoint, test_case_text) in enumerate(grouped_test_cases.items(), start=1):
            st.subheader(viewpoint)
            df = parse_test_cases_to_df(test_case_text)

            if df is not None:
                st.dataframe(df, use_container_width=True)
                # Thêm cột Viewpoint để phân biệt các Test Cases
                df["Viewpoint"] = viewpoint
                all_test_cases.append(df)

            else:
                st.text_area(f"Raw Output for {viewpoint}:", value=test_case_text, height=300, key=f"raw_output_{viewpoint}")

            # Cập nhật thanh tiến trình
            percent_complete = int((i / len(viewpoints)) * 100)
            progress_bar.progress(percent_complete)
            status_text.text(f"Progress: {percent_complete}%")

        # Khi hoàn thành, đổi thanh tiến trình thành màu xanh lá cây
        progress_bar.progress(100)
        status_text.text("✅ Completed!")
        st.success("✅ Test cases generated successfully!")

        # Gộp tất cả các Test Cases thành một DataFrame duy nhất
        if all_test_cases:
            combined_df = pd.concat(all_test_cases, ignore_index=True)

            # Tạo file CSV tổng hợp
            csv = combined_df.to_csv(index=False).encode("utf-8")

            # Nút tải xuống CSV tổng hợp
            st.download_button(
                "Download All Test Cases as CSV",
                csv,
                "all_test_cases.csv",
                "text/csv",
                key="download_all_test_cases"
            )