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
  private var model: VNCoreMLModel

  init() {
    let configuration = MLModelConfiguration()
    guard let mlModel = try? EmotionsImageClassifier(configuration: configuration).model else {
      fatalError("Failed to load model")
    }
    self.model = try! VNCoreMLModel(for: mlModel)
  }

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
        completion(nil, nil)
        return
      }

      let topResult = results.max(by: { a, b in a.confidence < b.confidence })
      completion(topResult?.identifier, topResult?.confidence)
    }

    let handler = VNImageRequestHandler(ciImage: ciImage)
    DispatchQueue.global(qos: .userInteractive).async {
      do {
        try handler.perform([request])
      } catch {
        completion(nil, nil)
      }
    }
  }

  func retrainModel(with image: UIImage, label: String, completion: @escaping (Bool) -> Void) {
    // Pseudo-code for retraining the model on-device
    // Actual implementation will vary based on tools like Core ML or Create ML
    DispatchQueue.global(qos: .background).async {
      // Perform retraining here
      let success = true  // Replace with actual success/failure of retraining
      completion(success)
    }
  }
}



