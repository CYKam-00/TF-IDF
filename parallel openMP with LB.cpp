#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono> 
#include "Documents.h"
#include <omp.h>

#define NUM_THREAD 4

//UPDATED: 24/9/2023 10.49pm

std::string preprocessWord(const std::string& word) {
    std::string cleanedWord = word;

    // Convert to lowercase
    for (char& c : cleanedWord) {
        c = std::tolower(c);
    }

    // Remove non-alphanumeric characters (e.g., punctuation)
    cleanedWord.erase(std::remove_if(cleanedWord.begin(), cleanedWord.end(), [](char c) {
        return !std::isalnum(c);
        }), cleanedWord.end());

    return cleanedWord;
}

// Tokenizes a string into individual words
std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string word;
    while (stream >> word) {
        std::string cleanedWord = preprocessWord(word);
        if (!cleanedWord.empty()) {
            tokens.push_back(cleanedWord);
        }
    }
    return tokens;
}

// Calculate term frequency for a document
std::unordered_map<std::string, double> computeTF(const std::unordered_map<std::string, int>& wordCount, int docLength) {
    std::unordered_map<std::string, double> tf;
    for (const auto& word : wordCount) {
        tf[word.first] = static_cast<double>(word.second) / docLength;
    }
    return tf;
}

// Calculate inverse document frequency for a word in the corpus
double computeIDF(int totalDocs, int docsContainingTerm) {
    return log10(static_cast<double>(totalDocs) / docsContainingTerm);
}

int main() {
    std::vector<std::string> documents = importedSampleDocuments_4;

    std::vector<std::unordered_map<std::string, int>> docWordCounts;
    std::unordered_map<std::string, int> globalWordCounts;
    std::vector<int> docSize;

    //make sure the vector is initialized before accessing
    docWordCounts.resize(documents.size());
    docSize.resize(documents.size());

    omp_set_num_threads(NUM_THREAD);

    // Start measuring time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Calculate word counts per document and global word counts
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < documents.size(); i++) {
        std::string doc = documents[i];
        std::unordered_map<std::string, int> wordCounts;
        auto tokens = tokenize(doc);
        docSize[i] = tokens.size();

        for (const auto& word : tokens) {
            wordCounts[word]++;
        }
        for (const auto pair : wordCounts)
        {
#pragma omp critical
            globalWordCounts[pair.first]++;
        }
        docWordCounts[i] = wordCounts;
    }

    std::unordered_map<std::string, double> idfMap;
    int totalDocs = documents.size();
    for (const auto& pair : globalWordCounts) {
        idfMap[pair.first] = computeIDF(totalDocs, pair.second);
    }

    // Calculate TF-IDF matrix
    std::vector<std::unordered_map<std::string, double>> tfidfDocs;
    tfidfDocs.resize(docWordCounts.size());
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < docWordCounts.size(); i++) {
        const auto docCount = docWordCounts.at(i);
        std::unordered_map<std::string, double> tfidf;
        auto tf = computeTF(docCount, docSize[i]);
        for (const auto& term : tf) {
            tfidf[term.first] = term.second * idfMap[term.first];
        }
#pragma omp critical
        tfidfDocs[i] = tfidf;
    }

    // Stop measuring time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Print TF-IDF matrix in matrix-like form
    std::cout << "TF-IDF Matrix:" << std::endl << "            ";
    for (const auto& term : globalWordCounts) {
        std::cout << std::setw(15) << term.first;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < tfidfDocs.size(); ++i) {
        std::cout << "Document " << (i + 1) << ": ";
        for (const auto& term : globalWordCounts) {
            std::cout << std::setw(15) << tfidfDocs[i][term.first];
        }
        std::cout << std::endl;
    }

    // Calculate the elapsed time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    long long elapsedMilliseconds = duration.count();

    // Print the elapsed time
    std::cout << "Serial Time elapsed: " << elapsedMilliseconds << " milliseconds" << std::endl;

    return 0;

}
