//PINTON MATTIA
#include "briscola/bow_classifier.hpp"

#include <iostream>
#include <filesystem>
/**
 * @brief Offline training entry point for the BoVW classifier.
 *
 * Loads the reference card templates, builds the visual vocabulary and
 * the augmented reference histograms, and writes both to disk. The
 * trained classifier is not used here: this binary is meant to be run
 * once per training set, and the resulting files are then loaded at
 * runtime by the application (KMeansBriscolaProvider::getBoVWClassifier).
 *
 * Command-line arguments:
 *   argv[1]  templatesDir   directory of reference images named
 *                           "<rank>-<suit>.JPG" (e.g. "3-spades.JPG")
 *   argv[2]  vocabularyOut  output path for the vocabulary file
 *   argv[3]  histogramsOut  output path for the histograms file
 *   argv[4]  K (optional)   number of visual words; see the note below
 *
 * Exit codes:
 *   0  success
 *   1  wrong arguments, training failure, or I/O failure
 */
int main(int argc, char** argv) {
    // Require at least the three mandatory paths. K is optional and falls
    // back to a default, so a call with only three arguments is valid and
    // produces a run with the default vocabulary size.
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
        << " <templatesDir> <vocabularyOut> <histogramsOut> [K]\n"
        << "Example: " << argv[0]
        << " data/Briscola_Trentine models/bow/vocab.yml models/bow/hist.yml 800\n";
        return 1;
    }

    // Paths are passed as std::filesystem::path so that they work
    // unchanged on any platform and so that the classifier, which
    // takes path objects, does not need string conversions at the call
    // site.
    const std::filesystem::path templatesDir = argv[1];
    const std::filesystem::path vocabularyOut = argv[2];
    const std::filesystem::path histogramsOut = argv[3];

    // Vocabulary size K: number of visual words.
    // Choosing K is a trade-off:
    //   - K too small: every word is shared by many augmented variants,
    //     the vocabulary cannot discriminate between cards, and the match
    //     is ambiguous.
    //   - K too large: every word is shared by very few variants, the
    //     vocabulary overfits the training set and does not generalize to
    //     query crops.
    // With 40 cards and 49 augmented variants per card (1960 histograms
    // total), my testing showed that K = 40 and K = 200 are too
    // small (accuracy 0/4 on the four test rounds), while K = 800 is the
    // best observed configuration (accuracy 1/4, with the correct suit on
    // two additional rounds). The default of 40 is a placeholder that
    // produces a nearly useless vocabulary; pass a real value on the
    // command line.
    const int K = (argc >= 5) ? std::stoi(argv[4]) : 40;

    try {
        briscola::BoWClassifier bow;

        // train() builds the vocabulary and the reference histograms from
        // the templates, generating augmented variants for each template.
        // The third argument caps the number of SIFT descriptors kept per
        // variant: 2000 is what i used to get the best result
        // (pratically all the features are kept)
        bow.train(templatesDir, K, 2000);

        // Create the output directories if they do not exist.
        // cv::FileStorage fails with a
        // "cannot open" error if the parent directory is missing.
        if (!vocabularyOut.parent_path().empty()) {
            std::filesystem::create_directories(vocabularyOut.parent_path());
        }
        if (!histogramsOut.parent_path().empty()) {
            std::filesystem::create_directories(histogramsOut.parent_path());
        }

        // Persist both files. If save() succeeds, the classifier is ready
        // to be loaded at runtime; if it fails, the exception propagates
        // and the process exits with code 1. Note that save() is not
        // atomic: if the process is killed between the two writes, the
        // on-disk state is inconsistent and load() will fail on the next
        // run. For an offline, rarely-run training step this is acceptable.
        bow.save(vocabularyOut, histogramsOut);

        std::cout << "Saved vocabulary to " << vocabularyOut << "\n"
                  << "Saved histograms to " << histogramsOut << "\n";
    } catch (const std::exception& e) {
        // Catch-all for any exception raised by train() or save():
        // missing templates directory, unreadable file, too few templates,
        // k-means failure, I/O error. The message is printed as-is so the
        // caller sees the full context (which file, which operation).
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}