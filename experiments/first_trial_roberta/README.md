### First Trial – RoBERTa

- **Overview**: A preliminary experiment using a pretrained model without fine-tuning. The dataset was preprocessed to closely follow the baseline setup in the original CROSSNEWS paper.
- **Dataset**: CROSSNEWS Gold + Merged Tweets
- **Number of Positive/Negative Pairs**: 1:1 balanced
- **Model**: Pretrained `roberta-base`
- **Accuracy**: ~50%
- **Observation**: The domain gap between articles and tweets appears to be too large. More advanced techniques such as prompt engineering or a more robust model may be required for effective cross-domain authorship verification.

**Next Step**: Apply prompt engineering to rewrite texts using LLMs.