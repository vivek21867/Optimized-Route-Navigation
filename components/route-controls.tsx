"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { MapPin, Navigation, BarChart3 } from "lucide-react"

interface RouteControlsProps {
  startPoint: { lat: number; lng: number }
  endPoint: { lat: number; lng: number }
  algorithm: "dijkstra" | "astar"
  setAlgorithm: (algorithm: "dijkstra" | "astar") => void
  trafficLevel: "low" | "medium" | "high"
  setTrafficLevel: (level: "low" | "medium" | "high") => void
  calculateRoute: () => void
  isCalculating: boolean
  distance: number | null
  time: number | null
}

export default function RouteControls({
  startPoint,
  endPoint,
  algorithm,
  setAlgorithm,
  trafficLevel,
  setTrafficLevel,
  calculateRoute,
  isCalculating,
  distance,
  time,
}: RouteControlsProps) {
  return (
    <Card className="shadow-md">
      <CardHeader className="pb-2">
        <CardTitle>Route Controls</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-xs font-medium text-gray-500">Start Point</Label>
              <div className="flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
                <MapPin className="h-4 w-4 text-green-500" />
                <span className="text-sm truncate">
                  {startPoint.lat.toFixed(4)}, {startPoint.lng.toFixed(4)}
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <Label className="text-xs font-medium text-gray-500">End Point</Label>
              <div className="flex items-center space-x-2 p-2 bg-gray-50 rounded-md">
                <MapPin className="h-4 w-4 text-red-500" />
                <span className="text-sm truncate">
                  {endPoint.lat.toFixed(4)}, {endPoint.lng.toFixed(4)}
                </span>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-500">Algorithm</Label>
            <RadioGroup
              value={algorithm}
              onValueChange={(value) => setAlgorithm(value as "dijkstra" | "astar")}
              className="flex space-x-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="dijkstra" id="dijkstra" />
                <Label htmlFor="dijkstra" className="text-sm">
                  Dijkstra
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="astar" id="astar" />
                <Label htmlFor="astar" className="text-sm">
                  A* Algorithm
                </Label>
              </div>
            </RadioGroup>
          </div>

          <div className="space-y-2">
            <Label className="text-xs font-medium text-gray-500">Traffic Level</Label>
            <RadioGroup
              value={trafficLevel}
              onValueChange={(value) => setTrafficLevel(value as "low" | "medium" | "high")}
              className="flex space-x-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="low" id="low" />
                <Label htmlFor="low" className="text-sm">
                  Low
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="medium" id="medium" />
                <Label htmlFor="medium" className="text-sm">
                  Medium
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="high" id="high" />
                <Label htmlFor="high" className="text-sm">
                  High
                </Label>
              </div>
            </RadioGroup>
          </div>

          <Button onClick={calculateRoute} disabled={isCalculating} className="w-full">
            {isCalculating ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                Calculating...
              </>
            ) : (
              <>
                <Navigation className="mr-2 h-4 w-4" />
                Calculate Route
              </>
            )}
          </Button>

          {distance !== null && time !== null && (
            <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-100">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-800">Route Summary</span>
                <BarChart3 className="h-4 w-4 text-blue-500" />
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <p className="text-gray-500 text-xs">Distance</p>
                  <p className="font-medium">{distance} km</p>
                </div>
                <div>
                  <p className="text-gray-500 text-xs">Est. Time</p>
                  <p className="font-medium">{Math.round(time)} min</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

