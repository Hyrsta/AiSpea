# 示例：多任务学习架构
class ChildAnalysisModel(nn.Module):
    def forward(self, dialog):
        # 主干网络
        shared_embed = TransformerEncoder(dialog)  
        
        # 多任务头
        lang_head = LanguageLevelClassifier(shared_embed)  # L1-L3分类
        emotion_head = EmotionTracker(shared_embed)        # 情感状态转移矩阵
        interest_head = InterestGraphGenerator(shared_embed) # 知识图谱节点激活
        
        return lang_head, emotion_head, interest_head
