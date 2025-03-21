"use client"

import { AlertCircle, CheckCircle, Info } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

interface ResultsDisplayProps {
  results: {
    prediction: string
    confidence: number
    description: string
    recommendations: string[]
  }
}

export default function ResultsDisplay({ results }: ResultsDisplayProps) {
  const { prediction, confidence, description, recommendations } = results

  // Determine severity level based on the condition and confidence
  const getSeverityLevel = () => {
    if (prediction === "Melanoma") {
      return {
        level: "High",
        color: "text-red-600",
        bgColor: "bg-red-100",
        icon: <AlertCircle className="h-5 w-5 text-red-600" />,
      }
    } else if (confidence > 0.9) {
      return {
        level: "Medium",
        color: "text-amber-600",
        bgColor: "bg-amber-100",
        icon: <Info className="h-5 w-5 text-amber-600" />,
      }
    } else {
      return {
        level: "Low",
        color: "text-green-600",
        bgColor: "bg-green-100",
        icon: <CheckCircle className="h-5 w-5 text-green-600" />,
      }
    }
  }

  const severity = getSeverityLevel()

  return (
    <Card className="h-full">
      <CardHeader className="pb-2">
        <CardTitle className="text-xl">Analysis Results</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-1">
              <h3 className="font-medium">Predicted Condition</h3>
              <span className="text-sm text-muted-foreground">{(confidence * 100).toFixed(1)}% confidence</span>
            </div>
            <Progress value={confidence * 100} className="h-2" />
            <div className="mt-2 flex items-center">
              <span className="text-lg font-semibold">{prediction}</span>
            </div>
          </div>

          <div className={`p-3 rounded-md ${severity.bgColor} flex items-start gap-2`}>
            {severity.icon}
            <div>
              <p className={`font-medium ${severity.color}`}>{severity.level} attention recommended</p>
              <p className="text-sm mt-1">
                {severity.level === "High"
                  ? "Please consult a healthcare professional as soon as possible."
                  : severity.level === "Medium"
                    ? "Consider scheduling an appointment with a dermatologist."
                    : "Monitor the condition and consult a doctor if it changes."}
              </p>
            </div>
          </div>

          <div>
            <h3 className="font-medium mb-1">About this condition</h3>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>

          <div>
            <h3 className="font-medium mb-2">Recommendations</h3>
            <ul className="space-y-1">
              {recommendations.map((rec, index) => (
                <li key={index} className="text-sm flex items-start gap-2">
                  <span className="bg-primary/10 text-primary rounded-full w-5 h-5 flex items-center justify-center flex-shrink-0 mt-0.5">
                    {index + 1}
                  </span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

