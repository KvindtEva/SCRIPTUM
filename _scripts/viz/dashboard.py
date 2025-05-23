# %%
import datasets
import turftopic
import topicwizard

# %%
# DOCUMENTS
ds = datasets.load_dataset("janko/scriptum")
documents = ds["train"]["cleaned_text"]

# %%
# MODEL
topmodel = turftopic.load_model("janko/s3_scriptum")
topmodel.encoder[0] = topmodel.encoder[0].half()
# topmodel.encoder = topmodel.encoder.to("cuda")

# %%
topic_data = topmodel.prepare_topic_data(
    corpus=documents,
    embeddings=topmodel.embeddings
    )

# %%
# topicwizard.visualize(topic_data=topic_data)

# %%
from topicwizard.figures import document_map

document_map(topic_data)
