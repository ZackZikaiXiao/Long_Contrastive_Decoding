import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import torch.nn.functional as F
from numpy.linalg import norm

# nltk.download('punkt')
cos_sim = lambda a, b: (a @ b.T) / (norm(a) * norm(b))

class TextProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.config = None

        if "jina" in model_path:
            # 如果是第二个模型，直接加载Jina模型
            self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)
            self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en")
            self.config = AutoConfig.from_pretrained("jinaai/jina-embeddings-v2-base-en")
        else:
            # 默认加载第一个模型
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)

    def split_text_into_segments(self, input_text, max_segment_length=512):
    # 根据逗号、冒号等常见标点符号直接拆分
        def further_split_text(text):
            split_chars = [',', ':', ';', '.', '!', '?']  # 可以根据需要添加更多标点符号
            for char in split_chars:
                text = text.replace(char, char + "###SPLIT###")  # 标记拆分点
            return text.split("###SPLIT###")

        # 直接分割整个文本
        sentences = further_split_text(input_text)

        # 如果没有指定最大长度，使用tokenizer的最大长度
        if max_segment_length is None:
            max_segment_length = self.config.max_position_embeddings

        segments = []
        current_segment = []

        for sentence in sentences:
            # 当前片段加上新句子后的长度
            current_segment_length = len(self.tokenizer.encode("".join(current_segment) + sentence, add_special_tokens=False))

            if current_segment_length <= max_segment_length:
                current_segment.append(sentence)
            else:
                # 如果加上当前句子超出最大长度，将当前片段加入结果并开始新的片段
                segments.append("".join(current_segment))
                current_segment = [sentence]

        # 如果还有剩余的句子
        if current_segment:
            segments.append("".join(current_segment))

        return segments

    
    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:, 0]

    def rank_documents_by_similarity(self, query, documents, max_segment_length=512):
        # if "jina" in self.model_path:
        #     # 如果使用的是第二个模型，使用Jina模型进行编码
        #     embeddings = self.model.encode([query] + documents, max_length=2048)
        #     query_embedding = embeddings[0]
        #     doc_embeddings = embeddings[1:]

        #     similarity_scores = [cos_sim(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]
        # else:
        # 将query和documents一起进行编码
        sentences = [query] + documents

        if max_segment_length is None:
            max_segment_length = self.config.max_position_embeddings
            
        # 对句子进行编码
        encoded_input = self.tokenizer(sentences, max_length=max_segment_length, padding=True, truncation=True, return_tensors='pt')

        # 计算token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 进行CLS池化
        sentence_embeddings = self.cls_pooling(model_output, encoded_input['attention_mask'])

        # 对嵌入进行归一化处理
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # 计算query和每个document的相似度
        query_embedding = sentence_embeddings[0]
        doc_embeddings = sentence_embeddings[1:]
        similarity_scores = torch.matmul(query_embedding, doc_embeddings.T)

        # 根据相似度分数从高到低排序文档和分数
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_documents = [documents[i] for i in sorted_indices]
        sorted_scores = [similarity_scores[i].item() for i in sorted_indices]

        return sorted_documents, sorted_scores

# 示例调用
if __name__ == "__main__":
    # 初始化类实例
    processor = TextProcessor(model_path='./models/jina-embeddings-v2-base-en')  # jina-embeddings-v2-base-en; dragon-plus-context-encoder
    
    # 文本分段
    input_text = """user\n\nGiven the JSON object below, extract and return only the value corresponding to the specified key.\n\nJSON data:\n{"14c78d4b-77fb-4235-afb2-311003fc675c": "76e7a462-4921-4dec-920e-f3f17e0cd8b6", "017e3b8d-a229-41b0-a564-0ffd6ebc83fe": "479226eb-4e4b-4ff4-8c87-71f322b23799", "755b0fc5-fd0c-4ea4-a454-40dd18f2db3c": "ba0c3b8f-d7db-4667-bf7c-0abb5c58c05a", "c5dfcae5-8e24-4218-a0af-9a984510e27b": "79e93bfc-3d68-4ecd-a275-d08bb52f852c", "15735254-4b2e-4718-9063-d6a92e..."""
    max_segment_length = 16
    documents = processor.split_text_into_segments(input_text, max_segment_length)

    print("分割后的片段:")
    for i, segment in enumerate(documents):
        print(f"Segment {i+1}:\n{segment}\n")
    
    query = "ec-920e-f3f"
    # 文档相似度排序
    # query = "how much protein should a female eat"
    # documents = [
    #     "The recommended daily protein intake for women is around 46 grams, according to health guidelines.",
    #     "For women who are active or pregnant, the protein requirement may be higher than the standard recommendation.",
    #     "Protein is essential for muscle repair and overall health, especially in women.",
    #     "Women who engage in regular exercise should consider increasing their protein intake to support muscle recovery.",
    #     "Dietary protein plays a crucial role in maintaining a healthy weight and body composition for women.",
    #     "A balanced diet rich in protein can help women meet their nutritional needs and maintain energy levels.",
    #     "Sources of protein like chicken, tofu, and beans are beneficial for women's health.",
    #     "Consuming enough protein is important, but women should also focus on getting a variety of nutrients.",
    #     "Protein is one of many nutrients that contribute to a healthy diet for women.",
    #     "A healthy diet includes not just protein but also carbohydrates, fats, vitamins, and minerals.",
    #     "how much protein should a female eat",
    #     "how protein should a female eat"
    # ]
    
    sorted_docs, sorted_scores = processor.rank_documents_by_similarity(query, documents)

    print("按照相似度排序的文档及对应分数:")
    for doc, score in zip(sorted_docs, sorted_scores):
        print(f"{score:.4f} - {doc}")
