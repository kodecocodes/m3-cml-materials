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
import Combine
import CoreML
import Vision

class EmotionDetectionViewModel: ObservableObject {
  @Published var image: UIImage?
  @Published var emotion: String?
  @Published var accuracy: String?
  @Published var isModelUpdating: Bool = false

  private let classifier: EmotionClassifier

  init(classifier: EmotionClassifier = EmotionClassifier()) {
    self.classifier = classifier
  }

  // Function to classify an image and update the emotion and accuracy properties
  func classifyImage() {
    if let image = self.image {
      let resizedImage = resizeImage(image)
      DispatchQueue.global(qos: .userInteractive).async {
        self.classifier.classify(image: resizedImage ?? image) { [weak self] emotion, confidence in
          DispatchQueue.main.async {
            self?.emotion = emotion ?? "Unknown"
            self?.accuracy = String(format: "%.2f%%", (confidence ?? 0) * 100.0)
          }
        }
      }
    }
  }

  // Reset the image, emotion, and accuracy to initial state
  func reset() {
    DispatchQueue.main.async {
      self.image = nil
      self.emotion = nil
      self.accuracy = nil
    }
  }

  // Resize the image to 224x224 pixels to match model input requirements
  private func resizeImage(_ image: UIImage) -> UIImage? {
    UIGraphicsBeginImageContext(CGSize(width: 224, height: 224))
    image.draw(in: CGRect(x: 0, y: 0, width: 224, height: 224))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return resizedImage
  }

  // MARK: - Personalizing Model Updates

  // New method to handle model updates with user-provided label
  func updateModel(with image: UIImage, label: String) {
    isModelUpdating = true
    DispatchQueue.global(qos: .userInitiated).async {
      self.executeUpdateTask(with: [(image, label)])
      DispatchQueue.main.async {
        self.isModelUpdating = false
      }
    }
  }

  // Execute the update task with provided samples
  func executeUpdateTask(with samples: [(UIImage, String)]) {
    guard let trainingData = prepareUpdateData(with: samples),
          let updateTask = createUpdateTask(with: trainingData) else {
      print("Failed to create update task.")
      return
    }

    updateTask.resume()
  }

  // Prepare the training data for model updates
  func prepareUpdateData(with samples: [(UIImage, String)]) -> MLBatchProvider? {
    var featureProviders = [MLFeatureProvider]()
    let inputName = "image"
    let outputName = "target"

    for sample in samples {
      guard let resizedImage = resizeImage(sample.0),
            let imageData = resizedImage.pngData() else { continue }

      let imageURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(UUID().uuidString + ".png")
      do {
        try imageData.write(to: imageURL)
      } catch {
        print("Failed to write image data to URL: \(error)")
        continue
      }

      guard let classifierConstraint = classifier.imageConstraint, let inputValue = try? MLFeatureValue(imageAt: imageURL, constraint: classifierConstraint) else { continue }
      let outputValue = MLFeatureValue(string: sample.1)

      let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue, outputName: outputValue]

      if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
        featureProviders.append(provider)
      }
    }

    return featureProviders.isEmpty ? nil : MLArrayBatchProvider(array: featureProviders)
  }

  // Create an update task for the model with the provided data
  func createUpdateTask(with data: MLBatchProvider) -> MLUpdateTask? {
    let modelURL = Bundle.main.url(forResource: "EmotionsImageClassifier", withExtension: "mlmodelc") ?? EmotionsImageClassifier.urlOfModelInThisBundle

    do {
      let updateTask = try MLUpdateTask(forModelAt: modelURL, trainingData: data, configuration: nil, completionHandler: { context in
        let updatedModel = context.model
        self.saveUpdatedModel(updatedModel)
      })
      return updateTask
    } catch {
      print("Error creating update task: \(error.localizedDescription)")
      print("Error details: \(error)")
      return nil
    }
  }


  // Save the updated model to the app's documents directory
  func saveUpdatedModel(_ updatedModel: MLModel) {
    let fileManager = FileManager.default
    let updatedModelURL = getDocumentsDirectory().appendingPathComponent("UpdatedModel.mlmodelc")

    do {
      let compiledModelURL = try MLModel.compileModel(at: updatedModelURL)
      if fileManager.fileExists(atPath: compiledModelURL.path) {
        _ = try fileManager.replaceItemAt(updatedModelURL, withItemAt: compiledModelURL)
      } else {
        try fileManager.moveItem(at: compiledModelURL, to: updatedModelURL)
      }
      print("Updated model saved to: \(updatedModelURL.path)")
    } catch {
      print("Could not save the updated model: \(error)")
    }
  }

  // Get the URL for the app's documents directory
  private func getDocumentsDirectory() -> URL {
    return FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
  }
}
