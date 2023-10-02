#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <chrono> // For measuring time
#include "Documents.h"

// Tokenizes a string into individual words
std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream stream(str);
    std::string word;
    while (stream >> word) {
        tokens.push_back(word);
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
    std::vector<std::unordered_map<std::string, int>> docWordCounts;
    std::unordered_map<std::string, int> globalWordCounts;
    std::vector<int> docSize;

    // convert all characters to lowercase
    std::vector<std::string> documents = importedSampleDocuments_4;

    for (auto& doc : documents) {
        for (char& c : doc) {
            c = std::tolower(c);
        }
    }


    // Start measuring time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Calculate word counts per document and global word counts
    int i = 0;
    for (const auto& doc : documents) {
        std::unordered_map<std::string, int> wordCounts;
        auto tokens = tokenize(doc);
        docSize.push_back(tokens.size());
        for (const auto& word : tokens) {
            wordCounts[word]++;
        }
        for (const auto pair : wordCounts)
        {
            globalWordCounts[pair.first]++;
        }
        docWordCounts.push_back(wordCounts);
        i++;
    }

    std::unordered_map<std::string, double> idfMap;
    int totalDocs = documents.size();
    for (const auto& pair : globalWordCounts) {
        idfMap[pair.first] = computeIDF(totalDocs, pair.second);
    }

    // Calculate TF-IDF matrix
    std::vector<std::unordered_map<std::string, double>> tfidfDocs;
    for (const auto& docCount : docWordCounts) {
        std::unordered_map<std::string, double> tfidf;
        auto tf = computeTF(docCount, docCount.size());
        for (const auto& term : tf) {
            tfidf[term.first] = term.second * idfMap[term.first];
        }
        tfidfDocs.push_back(tfidf);
    }

    // Stop measuring time
    auto endTime = std::chrono::high_resolution_clock::now();

    // Print TF-IDF matrix in matrix-like form
    std::cout << "TF-IDF Matrix:" << std::endl;
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
    std::cout << "total the: " << globalWordCounts["The"] << std::endl;

    return 0;

}
