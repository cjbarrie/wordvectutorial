# # shuffle text
# set.seed(123L)
# text <- sample(twts_sample$tweet)

twts_corpus <- corpus(twts_sample, text_field = "tweet")
toks <- tokens(twts_corpus)

# Get top k features
combined_dfm <- dfm(toks, verbose = TRUE)
top_feats <- featnames(combined_dfm)

# leave the pads so that non-adjacent words will not become adjacent
toks_feats <- tokens_select(toks, top_feats, padding = TRUE)

# Construct the feature co-occurrence matrix
toks_fcm <- fcm(
  toks_feats,
  context = "window",
  window = WINDOW_SIZE,
  count = "frequency",
  tri = FALSE,
  weights = rep(1, WINDOW_SIZE)
)

glove <- GlobalVectors$new(rank = DIM, x_max = 100, learning_rate = 0.05)
wv_main <- glove$fit_transform(
  toks_fcm,
  n_iter = ITERS,
  convergence_tol = 1e-3,
  n_threads = parallel::detectCores()
)
wv_context <- glove$components
glove_embedding <- wv_main + t(wv_context)

saveRDS(glove_embedding, file = "local_glove.rds")


# GloVe dimension reduction
glove_umap <- umap(glove_embedding, n_components = 2, 
                   metric = "cosine", n_neighbors = 25, 
                   min_dist = 0.1, spread=2, fast_sgd = TRUE)
save(glove_umap, file = "data/wordembed/glove_umap.RData")

# Put results in a dataframe for ggplot
df_glove_umap <- as.data.frame(glove_umap[["layout"]])

# Add the labels of the words to the dataframe
df_glove_umap$word <- rownames(df_glove_umap)
colnames(df_glove_umap) <- c("UMAP1", "UMAP2", "word")

# Plot the UMAP dimensions
ggplot(df_glove_umap) +
  geom_point(aes(x = UMAP1, y = UMAP2), colour = 'blue', size = 0.05) +
  ggplot2::annotate("rect", xmin = -3, xmax = -2, ymin = 5, ymax = 7,alpha = .2) +
  labs(title = "GloVe word embedding in 2D using UMAP")

# Plot the shaded part of the GloVe word embedding with labels
ggplot(df_glove_umap[df_glove_umap$UMAP1 < -2.5 & df_glove_umap$UMAP1 > -2.6 & df_glove_umap$UMAP2 > -2.5 & df_glove_umap$UMAP2 < -2.4,]) +
  geom_point(aes(x = UMAP1, y = UMAP2), colour = 'blue', size = 2) +
  geom_text(aes(UMAP1, UMAP2, label = word), size = 2.5, vjust=-1, hjust=0) +
  labs(title = "GloVe word embedding in 2D using UMAP - partial view") +
  theme(plot.title = element_text(hjust = .5, size = 14))


# Plot the word embedding of words that are related for the GloVe model
word <- glove_embedding["economy",, drop = FALSE]
cos_sim = sim2(x = glove_embedding, y = word, method = "cosine", norm = "l2")
select <- data.frame(rownames(as.data.frame(head(sort(cos_sim[,1], decreasing = TRUE), 25))))
colnames(select) <- "word"
selected_words <- df_glove_umap %>% 
  inner_join(y=select, by= "word")

#The ggplot visual for GloVe
ggplot(selected_words, aes(x = UMAP1, y = UMAP2)) + 
  geom_point(show.legend = FALSE) + 
  geom_text(aes(UMAP1, UMAP2, label = word), show.legend = FALSE, size = 2.5, vjust=-1.5, hjust=0) +
  labs(title = "GloVe word embedding of words related to 'economy'") +
  theme(plot.title = element_text(hjust = .5, size = 14))









first_index <- which(row.names(glove_embedding) == "OpenAI")
second_index <- which(row.names(glove_embedding) == "crime")

nn_first <- RANN::nn2(glove_embedding, 
                        glove_embedding[first_index, , drop = FALSE], 
                        k = 250)$nn.idx

nn_second <- RANN::nn2(glove_embedding, 
                      glove_embedding[second_index, , drop = FALSE], 
                      k = 250)$nn.idx


# Put results in a dataframe for ggplot
df_glove_umap <- as.data.frame(glove_umap[["layout"]])

# Add the labels of the words to the dataframe
df_glove_umap$word <- rownames(df_glove_umap)
colnames(df_glove_umap) <- c("UMAP1", "UMAP2", "word")

df_glove_umap$Category <- "Third"
df_glove_umap$Category[nn_first] <- "First"
df_glove_umap$Category[nn_second] <- "Second"

ggplot(df_glove_umap, aes(x = UMAP1, y = UMAP2, color = Category, alpha = Category)) +
  geom_point() +
  scale_alpha_manual(values = c("Economy" = 1, "Crime" = 1, "Third" = 0.01)) +
  theme_minimal() +
  guides(alpha = "none") +  # This line removes the alpha legend
  labs(title = "Word Embeddings with Nearest Neighbors Highlighted")
