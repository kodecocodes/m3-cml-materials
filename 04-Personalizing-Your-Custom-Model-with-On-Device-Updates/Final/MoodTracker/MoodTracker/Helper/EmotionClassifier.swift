/// Copyright (c) 2024 Kodeco Inc.
///
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
///
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
///
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
///
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import SwiftUI
import Vision
import CoreML

class EmotionClassifier {
    private(set) var emotionsImageClassifier: EmotionsImageClassifier
    private(set) var model: VNCoreMLModel

    /// Returns the image constraint for the model's "inputFeature" input feature.
    var imageConstraint: MLImageConstraint? {
        let description = emotionsImageClassifier.model.modelDescription

        // Ensure this matches the actual input feature name in your model
        let inputName = "image"
        guard let imageInputDescription = description.inputDescriptionsByName[inputName] else {
            print("Input feature \(inputName) not found in model description.")
            return nil
        }

        return imageInputDescription.imageConstraint
    }

    init() {
        let configuration = MLModelConfiguration()
        if let classifier = EmotionClassifier.loadUpdatedModel() {
            self.emotionsImageClassifier = classifier
            self.model = try! VNCoreMLModel(for: classifier.model)
        } else {
            guard let classifier = try? EmotionsImageClassifier(configuration: configuration) else {
                fatalError("Failed to load model")
            }
            self.emotionsImageClassifier = classifier
            self.model = try! VNCoreMLModel(for: classifier.model)
        }
    }

    // Classify the image and return the result in a completion handler
    func classify(image: UIImage, completion: @escaping (String?, Float?) -> Void) {
        guard let ciImage = CIImage(image: image) else {
            completion(nil, nil)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("Error during classification: \(error.localizedDescription)")
                completion(nil, nil)
                return
            }

            guard let results = request.results as? [VNClassificationObservation] else {
                print("No results found")
                completion(nil, nil)
                return
            }

            let topResult = results.max(by: { a, b in a.confidence < b.confidence })
            guard let bestResult = topResult else {
                print("No top result found")
                completion(nil, nil)
                return
            }

            completion(bestResult.identifier, bestResult.confidence)
        }

        let handler = VNImageRequestHandler(ciImage: ciImage)

        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([request])
            } catch {
                print("Failed to perform classification: \(error.localizedDescription)")
                completion(nil, nil)
            }
        }
    }

    // Load the updated model from the documents directory if available
    static func loadUpdatedModel() -> EmotionsImageClassifier? {
        let updatedModelURL = getDocumentsDirectory().appendingPathComponent("UpdatedModel.mlmodelc")
        guard FileManager.default.fileExists(atPath: updatedModelURL.path),
              let updatedModel = try? EmotionsImageClassifier(contentsOf: updatedModelURL) else {
            print("Failed to load the updated model.")
            return nil
        }

        return updatedModel
    }

    // Get the URL for the app's documents directory
    private static func getDocumentsDirectory() -> URL {
        return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }
}
