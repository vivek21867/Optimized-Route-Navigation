"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Upload, ImagePlus } from "lucide-react"
import ResultsDisplay from "./results-display"

export default function UploadForm() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<null | {
    prediction: string
    confidence: number
    description: string
    recommendations: string[]
  }>(null)

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        setSelectedImage(reader.result as string)
        setResults(null)
      }
      reader.readAsDataURL(file)
    }
  }

  const analyzeImage = () => {
    if (!selectedImage) return

    setIsAnalyzing(true)

    // Simulate API call to backend model
    setTimeout(() => {
      // Mock results - in a real app, this would come from your trained model
      const mockResults = [
        {
          prediction: "Melanoma",
          confidence: 0.87,
          description:
            "Melanoma is a type of skin cancer that can be serious because it has a high risk of spreading to other parts of the body if not detected and treated early.",
          recommendations: [
            "Consult a dermatologist immediately",
            "Avoid sun exposure to the affected area",
            "Document any changes in the lesion",
          ],
        },
        {
          prediction: "Eczema",
          confidence: 0.92,
          description:
            "Eczema (atopic dermatitis) is a condition that makes your skin red and itchy. It's common in children but can occur at any age.",
          recommendations: [
            "Keep the skin moisturized",
            "Avoid known triggers",
            "Consider over-the-counter hydrocortisone cream",
          ],
        },
        {
          prediction: "Psoriasis",
          confidence: 0.78,
          description:
            "Psoriasis is a skin disorder that causes skin cells to multiply up to 10 times faster than normal, resulting in bumpy red patches covered with white scales.",
          recommendations: [
            "Moisturize regularly",
            "Avoid triggers like stress",
            "Consider light therapy or prescribed medications",
          ],
        },
        {
          prediction: "Acne",
          confidence: 0.95,
          description:
            "Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells, leading to whiteheads, blackheads or pimples.",
          recommendations: [
            "Wash affected areas twice daily",
            "Avoid touching or picking at the area",
            "Consider over-the-counter acne products with benzoyl peroxide",
          ],
        },
      ]

      // Randomly select one of the mock results
      const randomIndex = Math.floor(Math.random() * mockResults.length)
      setResults(mockResults[randomIndex])
      setIsAnalyzing(false)
    }, 2000)
  }

  return (
    <div>
      <div className="flex flex-col md:flex-row gap-6">
        <div className="flex-1">
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-4 h-64 flex flex-col items-center justify-center cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => document.getElementById("image-upload")?.click()}
          >
            {selectedImage ? (
              <img
                src={selectedImage || "/placeholder.svg"}
                alt="Selected skin"
                className="max-h-full max-w-full object-contain"
              />
            ) : (
              <>
                <ImagePlus className="h-12 w-12 text-gray-400 mb-3" />
                <p className="text-gray-500">Click to upload an image</p>
                <p className="text-xs text-gray-400 mt-1">PNG, JPG or JPEG</p>
              </>
            )}
            <input
              id="image-upload"
              type="file"
              accept="image/png, image/jpeg, image/jpg"
              className="hidden"
              onChange={handleImageChange}
            />
          </div>

          <div className="mt-4 flex justify-center">
            <Button onClick={analyzeImage} disabled={!selectedImage || isAnalyzing} className="w-full md:w-auto">
              {isAnalyzing ? (
                <>
                  <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <Upload className="mr-2 h-4 w-4" />
                  Analyze Image
                </>
              )}
            </Button>
          </div>
        </div>

        <div className="flex-1">
          {results ? (
            <ResultsDisplay results={results} />
          ) : (
            <Card className="h-full flex items-center justify-center p-6 bg-gray-50">
              <p className="text-gray-500 text-center">
                {isAnalyzing
                  ? "Analyzing your image..."
                  : selectedImage
                    ? "Click 'Analyze Image' to get results"
                    : "Upload an image to see analysis results"}
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

