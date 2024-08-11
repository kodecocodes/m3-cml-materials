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

class EmotionDetectionViewModel: ObservableObject {
  @Published var image: UIImage?
  @Published var emotion: String?
  @Published var accuracy: String?

  private let classifier = EmotionClassifier()

  func classifyImage() {
    if let image = self.image {
      // Resize the image before classification
      let resizedImage = resizeImage(image)
      DispatchQueue.global(qos: .userInteractive).async {
        self.classifier.classify(image: resizedImage ?? image) { [weak self] emotion, confidence in
          // Update the published properties on the main thread
          DispatchQueue.main.async {
            self?.emotion = emotion ?? "Unknown"
            self?.accuracy = String(format: "%.2f%%", (confidence ?? 0) * 100.0)
          }
        }
      }
    }
  }

  func reset() {
    DispatchQueue.main.async {
      self.image = nil
      self.emotion = nil
      self.accuracy = nil
    }
  }

  private func resizeImage(_ image: UIImage) -> UIImage? {
    UIGraphicsBeginImageContext(CGSize(width: 224, height: 224))
    image.draw(in: CGRect(x: 0, y: 0, width: 224, height: 224))
    let resizedImage = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return resizedImage
  }
}

