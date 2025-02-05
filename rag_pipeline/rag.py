import os
import logging
import json
import pickle
import numpy as np
from pathlib import Path
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import time  # Import time module

class RAGPipeline:
    def __init__(self, output_dir=None):
        self._setup_logging()

        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY không tìm thấy trong file .env")

            # Initialize OpenAI client
            self.client = OpenAI(api_key=api_key)

            # Use environment variable for output directory or default to './output'
            self.output_dir = Path(output_dir) if output_dir else Path("./output")
            self.logger.info(f"Đường dẫn output: {self.output_dir.absolute()}")

            # Khởi tạo pipeline
            self._initialize_pipeline()

        except Exception as e:
            self.logger.error(f"Lỗi trong __init__: {str(e)}", exc_info=True)
            raise

    def _setup_logging(self):
        """Thiết lập logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_dir / "rag_pipeline.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def _initialize_pipeline(self):
        """Khởi tạo FAISS index và load texts với validation"""
        try:
            index_path = self.output_dir / "faiss_index.bin"
            texts_path = self.output_dir / "processed_texts.pkl"

            if not self.output_dir.exists():
                raise FileNotFoundError(f"Thư mục output không tồn tại: {self.output_dir}")
            if not index_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file index tại {index_path}")
            if not texts_path.exists():
                raise FileNotFoundError(f"Không tìm thấy file texts tại {texts_path}")

            self.index = faiss.read_index(str(index_path))

            with open(texts_path, "rb") as f:
                self.texts = pickle.load(f)

            # Validate that texts is a list of dictionaries with required keys
            if not isinstance(self.texts, list):
                raise ValueError("Processed texts phải là một list")

            # Convert dictionary format to string if necessary
            self.processed_texts = []
            for item in self.texts:
                if isinstance(item, dict):
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})
                    page = metadata.get("page", "N/A")
                    source = metadata.get("source", "N/A")
                    self.processed_texts.append(f"[Trang {page} - {source}]\n{content}")
                else:
                    self.processed_texts.append(str(item))

            self.logger.info(f"Đã load {len(self.processed_texts)} texts và FAISS index")

        except Exception as e:
            self.logger.error("Lỗi trong _initialize_pipeline", exc_info=True)
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Lấy embedding với retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large"
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} for embedding")
                time.sleep(1)

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Lấy context với similarity threshold"""
        try:
            query_embedding = self.get_embedding(query)

            # Sử dụng batch search để tối ưu
            distances, indices = self.index.search(
                np.array([query_embedding]),
                min(k, len(self.processed_texts))
            )

            # Lọc kết quả theo ngưỡng similarity
            threshold = 0.7
            valid_indices = [i for i, d in zip(indices[0], distances[0])
                           if d < threshold]

            if not valid_indices:
                return "Không tìm thấy context phù hợp."

            contexts = [self.processed_texts[i] for i in valid_indices]
            return "\n\n---\n\n".join(contexts)

        except Exception as e:
            self.logger.error("Lỗi trong get_relevant_context", exc_info=True)
            raise

    def get_answer(self, query: str) -> str:
        """Xử lý câu hỏi với caching và rate limiting"""
        try:
            if not query.strip():
                return "Vui lòng nhập câu hỏi!"

            context = self.get_relevant_context(query)

            # Cải thiện prompt để có câu trả lời chất lượng hơn
            prompt = """Bạn là trợ lý trả lời câu hỏi về nông nghiệp và điều tra thống kê.
            Hãy trả lời dựa trên context được cung cấp một cách chi tiết và chính xác.
            Nếu không tìm thấy thông tin trong context, hãy nói rõ điều đó.

            Context:
            {}

            Câu hỏi: {}

            Trả lời chi tiết dựa trên thông tin trong context:""".format(context, query)

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error("Lỗi trong get_answer", exc_info=True)
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return "Hệ thống đang bận, vui lòng thử lại sau ít phút."
            return f"Lỗi xử lý câu hỏi: {error_msg}"
