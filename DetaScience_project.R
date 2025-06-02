# Required libraries
install.packages("rvest")
install.packages("dplyr")
install.packages("stringr")
install.packages("tm")
install.packages("SnowballC")
install.packages("qdapRegex")
install.packages("hunspell")
install.packages("topicmodels")
install.packages("tidytext")
install.packages("textclean")
install.packages("quanteda")

library(rvest)
library(dplyr)
library(tidytext)
library(stringr)
library(tm)
library(SnowballC)  # Load the SnowballC library
library(hunspell)
library(textclean)
library(topicmodels)
library(quanteda)

# Web scraping function
scrape_article <- function(url) {
  webpage <- read_html(url)
  
  # Extract article text
  article_text <- webpage %>%
    html_nodes("p") %>%
    html_text() %>%
    paste(collapse = " ")
  
  return(article_text)
}

# URL of the article
url <- "https://education.nationalgeographic.org/resource/air-pollution/"
raw_text <- scrape_article(url)

# 1. Text Cleaning
clean_text <- function(text) {
  text <- tolower(text)  # Convert to lowercase
  text <- str_replace_all(text, "[^[:alnum:]\\s]", " ")  # Remove special characters
  text <- str_replace_all(text, "\\d+", " ")  # Remove numbers
  text <- str_replace_all(text, "\\s+", " ")  # Remove extra whitespace
  text <- str_trim(text)  # Trim leading and trailing whitespace
  return(text)
}

# 2. Tokenization
tokenize_text <- function(text) {
  tokens <- unnest_tokens(data.frame(text = text), word, text)
  return(tokens)
}

# 3. Normalization (included in clean_text function)

# 4. Stop words Removal
remove_stopwords <- function(tokens) {
  tokens %>%
    anti_join(stop_words, by = "word")
}

# 5. Stemming
stem_words <- function(words) {
  wordStem(words, language = "english")
}

# 6. Handle Contractions
handle_contractions <- function(text) {
  replace_contraction(text)
}

# 7. Handle Emojis (remove them for simplicity)
remove_emojis <- function(text) {
  str_replace_all(text, "[^[:ascii:]]", "")
}

# 8. Spell Checking
spell_check <- function(words) {
  corrected_words <- sapply(words, function(word) {
    suggestions <- hunspell_suggest(word)
    if(length(suggestions) > 0 && !hunspell_check(word)) {
      return(suggestions[[1]][1])
    }
    return(word)
  })
  return(corrected_words)
}

# Process the text
processed_text <- raw_text %>%
  handle_contractions() %>%
  remove_emojis() %>%
  clean_text() %>%
  tokenize_text() %>%
  remove_stopwords()

# Apply stemming
processed_text$stemmed_word <- stem_words(processed_text$word)

# Apply spell checking
processed_text$corrected_word <- spell_check(processed_text$word)

# Save processed dataset
write.csv(processed_text, "processed_text_data.csv", row.names = FALSE)

# Display first few rows of processed data
head(processed_text)

# Step 1: Data Preparation (using already processed data)
# Load the processed data if it's not in memory
processed_data <- read.csv("processed_text_data.csv")

# Add document ID column
processed_data$doc_id <- 1  # Since all words are from one document

# Step 2: Creating Document-Term Matrix (DTM)
# Convert processed data to document-term matrix format
dtm <- processed_data %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

# Step 3: Calculate TF-IDF
tfidf <- weightTfIdf(dtm)
print(tfidf)

# Step 4: Topic Modeling with LDA
# Set random seed for reproducibility
set.seed(123)
# Specify number of topics (you can adjust this)
k <- 5
lda_model <- LDA(dtm, k = k, method = "Gibbs", 
                 control = list(iter = 2000, verbose = 25))

# Step 5: Examine the Topics
# Step 5.1: Get the most probable words for each topic
top_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  arrange(topic, -beta)

# Print top terms for each topic
print("Most probable words for each topic:")
print(top_terms)

# Step 5.2: Get topic proportions for the document
doc_topics <- tidy(lda_model, matrix = "gamma") %>%
  arrange(document, -gamma)

# Print document-topic distributions
print("Topic proportions for the document:")
print(doc_topics)

# Step 6: Visualize the Results
# Create a visualization of top terms per topic
library(ggplot2)

top_terms %>%
  ggplot(aes(reorder(term, beta), beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Terms per Topic",
       x = "Term",
       y = "Beta")

# Save the plot
ggsave("topic_model_visualization.png")

# Save results to files
# Combine top terms and document-topic distributions
# Rename columns for clarity
top_terms <- top_terms %>%
  rename(term_beta = beta)

doc_topics <- doc_topics %>%
  rename(doc_gamma = gamma)

# Create a unified data frame
combined_results <- top_terms %>%
  left_join(doc_topics, by = "topic") %>%
  select(
    topic,              # Topic number
    term,               # Term from top terms
    term_beta,          # Probability of term for the topic
    document,           # Document ID from document-topic distribution
    doc_gamma           # Probability of the topic for the document
  )

# Save the combined results to a single CSV file
write.csv(combined_results, "combined_topic_analysis.csv", row.names = FALSE)

# Display the first few rows of the combined results
head(combined_results)


# Create a comprehensive dataframe with all processing results
comprehensive_results <- processed_data %>%
  # Add topic probabilities for each word
  left_join(
    tidy(lda_model, matrix = "beta") %>%
      group_by(term) %>%
      mutate(dominant_topic = topic[which.max(beta)],
             topic_probability = max(beta)) %>%
      select(term, dominant_topic, topic_probability) %>%
      distinct(),
    by = c("word" = "term")
  )

# Add descriptive topic labels (optional)
topic_labels <- paste("Topic", 1:k)
comprehensive_results$topic_label <- topic_labels[comprehensive_results$dominant_topic]

# Organize columns in a meaningful order
final_dataset <- comprehensive_results %>%
  select(
    word,                  # Original word
    stemmed_word,         # Stemmed version
    corrected_word,       # Spell-checked version
    dominant_topic,       # Main topic number
    topic_label,          # Topic label
    topic_probability     # Probability of word belonging to topic
  ) %>%
  # Sort by topic label and probability (descending)
  arrange(topic_label, desc(topic_probability)) %>%
  # Add a rank within each topic based on probability
  group_by(topic_label) %>%
  mutate(rank_in_topic = row_number()) %>%
  ungroup()

# Add a blank line between topics for better readability
final_dataset_with_spaces <- final_dataset %>%
  group_by(topic_label) %>%
  do(add_row(., 
             word = "",
             stemmed_word = "",
             corrected_word = "",
             dominant_topic = NA,
             topic_label = "",
             topic_probability = NA,
             rank_in_topic = NA,
             .after = nrow(.))) %>%
  ungroup()

# Save the beautifully formatted dataset
write.csv(final_dataset_with_spaces, "comprehensive_text_analysis.csv", row.names = FALSE)

# Display the first few rows of the final dataset
head(final_dataset)
