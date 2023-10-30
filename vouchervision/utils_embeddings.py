from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')
instruction = "Represent a row from a table  describing and herbarium specimen."
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
