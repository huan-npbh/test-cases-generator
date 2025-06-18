# Tool: Generate Test Cases from Spec using Claude API + Streamlit UI
# Requirement: pip install pandas streamlit requests

import pandas as pd
import streamlit as st
import io
import re  # Thêm import re để sử dụng regex
import requests  # Thêm import requests để gọi Claude API
import anthropic  # Thêm import anthropic để sử dụng thư viện chính thức của Claude

# ====== CONFIGURATION ======
CLAUDE_API_KEY = "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"  # Thay bằng API Key thực tế của bạn
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)  # Tạo client Claude

# ====== FUNCTION TO GENERATE TEST CASES ======
def generate_test_cases(spec_text):
    prompt = f"""
    You are a QA test analyst.
    Given the following system specification, generate a test case table.
    Use this format:
    | Test Case ID | Description | Step | Expected Output | Priority |

    Specification:
    {spec_text}
    """
    response = client.completions.create(
        model="claude-3-5-haiku-20241022",
        max_tokens_to_sample=50,
        prompt=prompt,  # Cung cấp tham số 'prompt'
    )
    return response["completion"]

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

        # Xử lý cột "Step" để mỗi bước xuống dòng
        if "Step" in df.columns:
            df["Step"] = df["Step"].apply(
                lambda x: re.sub(r"^\s*(\d+\.)", r"\1", x).replace(". ", ".\n") if isinstance(x, str) else x
            )

        return df
    except Exception as e:
        st.error("Failed to parse table. Showing raw output.")
        return None

# ====== FUNCTION TO GENERATE TEST VIEWPOINTS ======
def generate_test_viewpoints(spec_text):
    prompt = f"""
    You are a QA test analyst.
    Given the following system specification, generate a list of test viewpoints.
    Test viewpoints are high-level testing perspectives or scenarios derived from the specification.
    Provide the viewpoints in a numbered list format.
    Just generate the test viewpoint without any additional text.

    Specification:
    {spec_text}
    """
    response = client.completions.create(
        model="claude-3-5-haiku-20241022",
        max_tokens_to_sample=50,
        prompt=prompt,  # Cung cấp tham số 'prompt'
    )
    return response["completion"]

# ====== FUNCTION TO GENERATE TEST CASES BY VIEWPOINT ======
def generate_test_cases_by_viewpoint(viewpoints_text):
    grouped_test_cases = {}
    viewpoints = [vp.strip() for vp in viewpoints_text.split("\n") if vp.strip()]

    for i, viewpoint in enumerate(viewpoints, start=1):
        prompt = f"""
        You are a QA test analyst.
        Given the following test viewpoint, generate a test case table.
        Just generate the test case table without any additional text.
        Use this format:
        | Test Case ID | Description | Step | Expected Output | Priority |

        Test Viewpoint:
        {viewpoint}
        """
        response = client.completions.create(
            model="claude-3-5-haiku-20241022",
            max_tokens_to_sample=50,
            prompt=prompt,  # Cung cấp tham số 'prompt'
        )
        grouped_test_cases[f"Viewpoint {viewpoint}"] = response["completion"]

    return grouped_test_cases

# ====== STREAMLIT UI ======
st.title("🧪 Test Case Generator from Spec AI Agent")

# Nhập đặc tả hệ thống
spec_input = st.text_area("Enter system specification:", height=200, key="spec_input")

# Biến lưu trữ Test Viewpoints
viewpoints_text = st.session_state.get("viewpoints_text", "")

# Nút Generate Test Viewpoints
if st.button("Generate Test Viewpoints"):
    if not CLAUDE_API_KEY or CLAUDE_API_KEY == "YOUR_CLAUDE_API_KEY_HERE":
        st.error("❗ Please update your Claude API key in the script.")
    elif not spec_input.strip():
        st.warning("Please enter a system specification.")
    else:
        with st.spinner("Generating test viewpoints, please wait..."):
            viewpoints_text = generate_test_viewpoints(spec_input)
            st.session_state["viewpoints_text"] = viewpoints_text  # Lưu vào session state
            st.success("✅ Test viewpoints generated successfully!")
            st.text_area("Test Viewpoints:", value=viewpoints_text, height=300, key="viewpoints_output")

# Hiển thị Test Viewpoints nếu đã tạo
if viewpoints_text:
    st.subheader("Generated Test Cases from Viewpoints")
    st.text_area("Test Viewpoints:", value=viewpoints_text, height=300, key="viewpoints_display")

    # Nút Generate Test Cases từ Test Viewpoints
    if st.button("Generate Test Cases from Viewpoints"):
        with st.spinner("Generating test cases, please wait..."):
            viewpoints = [vp.strip() for vp in viewpoints_text.split("\n") if vp.strip()]
            progress_bar = st.progress(0)  # Khởi tạo thanh tiến trình
            status_text = st.empty()  # Tạo vùng trống để hiển thị trạng thái
            grouped_test_cases = {}

            for i, viewpoint in enumerate(viewpoints, start=1):
                prompt = f"""
                You are a QA test analyst.
                Given the following test viewpoint, generate a test case table.
                Use this format:
                | Test Case ID | Description | Step | Expected Output | Priority |

                Test Viewpoint:
                {viewpoint}
                """
                headers = {
                    "Authorization": f"Bearer {CLAUDE_API_KEY}",
                    "Content-Type": "application/json",
                }
                data = {
                    "prompt": prompt,
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens_to_sample": 1000,
                    "stop_sequences": ["\n\n"],
                }
                response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
                response.raise_for_status()
                content = response.json()["completion"]
                grouped_test_cases[f"Viewpoint {viewpoint}"] = content

                # Cập nhật thanh tiến trình và trạng thái
                percent = int((i / len(viewpoints)) * 100)
                progress_bar.progress(percent)
                status_text.text(f"Progress: {percent}%")

            # Khi hoàn thành, đổi thanh tiến trình thành màu xanh lá cây
            progress_bar.progress(100)
            status_text.text("✅ Completed!")
            st.success("✅ Test cases generated successfully!")

        # Hiển thị Test Cases theo từng Viewpoint
        for viewpoint, test_case_text in grouped_test_cases.items():
            st.subheader(viewpoint)
            df = parse_test_cases_to_df(test_case_text)

            if df is not None:
                st.dataframe(df, use_container_width=True)

                # Nút tải xuống CSV cho từng Viewpoint
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(f"Download Test Cases for {viewpoint}", csv, f"{viewpoint.replace(':', '').replace(' ', '_')}_test_cases.csv", "text/csv", key=f"download_{viewpoint}")
            else:
                st.text_area(f"Raw Output for {viewpoint}:", value=test_case_text, height=300, key=f"raw_output_{viewpoint}")
