"use client"

import { useState } from "react"
import dynamic from "next/dynamic"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import AlgorithmExplanation from "@/components/algorithm-explanation"
import RouteControls from "@/components/route-controls"

// Dynamically import the Map component to avoid SSR issues with Leaflet
const MapWithNoSSR = dynamic(() => import("@/components/map"), {
  ssr: false,
  loading: () => (
    <div className="h-[600px] w-full flex items-center justify-center bg-gray-100 rounded-md">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
    </div>
  ),
})

export default function Home() {
  const [startPoint, setStartPoint] = useState({ lat: 40.7128, lng: -74.006 }) // New York
  const [endPoint, setEndPoint] = useState({ lat: 40.7614, lng: -73.9776 }) // Manhattan
  const [algorithm, setAlgorithm] = useState<"dijkstra" | "astar">("dijkstra")
  const [trafficLevel, setTrafficLevel] = useState<"low" | "medium" | "high">("medium")
  const [isCalculating, setIsCalculating] = useState(false)
  const [path, setPath] = useState<Array<[number, number]>>([])
  const [distance, setDistance] = useState<number | null>(null)
  const [time, setTime] = useState<number | null>(null)

  const calculateRoute = () => {
    setIsCalculating(true)

    // Simulate API call delay
    setTimeout(() => {
      // Generate a path between the two points with some randomness to simulate different routes
      const newPath = generatePath(startPoint, endPoint, algorithm, trafficLevel)
      setPath(newPath)

      // Calculate approximate distance in km
      const calculatedDistance = calculateDistance(newPath)
      setDistance(calculatedDistance)

      // Calculate approximate time based on distance and traffic
      const trafficMultiplier = trafficLevel === "low" ? 1 : trafficLevel === "medium" ? 1.5 : 2.2
      const calculatedTime = (calculatedDistance / 50) * 60 * trafficMultiplier // 50 km/h average speed
      setTime(calculatedTime)

      setIsCalculating(false)
    }, 1500)
  }

  return (
    <main className="min-h-screen p-4 md:p-8 bg-gray-50">
      <div className="container mx-auto max-w-7xl">
        <h1 className="text-3xl font-bold mb-2 text-primary">Optimized Route Navigation</h1>
        <p className="text-gray-600 mb-6">
          Visualize and compare routes using Dijkstra's algorithm with traffic considerations
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <Card className="shadow-md">
              <CardHeader className="pb-2">
                <CardTitle>Interactive Map</CardTitle>
                <CardDescription>Click on the map to set start and end points</CardDescription>
              </CardHeader>
              <CardContent>
                <MapWithNoSSR
                  startPoint={startPoint}
                  endPoint={endPoint}
                  path={path}
                  setStartPoint={setStartPoint}
                  setEndPoint={setEndPoint}
                />
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            <RouteControls
              startPoint={startPoint}
              endPoint={endPoint}
              algorithm={algorithm}
              setAlgorithm={setAlgorithm}
              trafficLevel={trafficLevel}
              setTrafficLevel={setTrafficLevel}
              calculateRoute={calculateRoute}
              isCalculating={isCalculating}
              distance={distance}
              time={time}
            />

            <Tabs defaultValue="algorithm" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="algorithm">Algorithm</TabsTrigger>
                <TabsTrigger value="traffic">Traffic Impact</TabsTrigger>
              </TabsList>
              <TabsContent value="algorithm" className="mt-2">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle>Dijkstra's Algorithm</CardTitle>
                    <CardDescription>How the pathfinding works</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <AlgorithmExplanation algorithm={algorithm} />
                  </CardContent>
                </Card>
              </TabsContent>
              <TabsContent value="traffic" className="mt-2">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle>Traffic Considerations</CardTitle>
                    <CardDescription>How traffic affects route selection</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-600">
                      Traffic conditions significantly impact route optimization. In high traffic:
                    </p>
                    <ul className="list-disc list-inside text-sm text-gray-600 mt-2 space-y-1">
                      <li>Edge weights increase proportionally to congestion</li>
                      <li>Alternative routes become more favorable</li>
                      <li>Travel time estimates adjust dynamically</li>
                    </ul>
                    <div className="mt-4 p-3 bg-amber-50 rounded-md border border-amber-200">
                      <p className="text-xs text-amber-800">
                        Current traffic level: <span className="font-semibold">{trafficLevel.toUpperCase()}</span>
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </main>
  )
}

// Helper function to generate a path between two points
function generatePath(
  start: { lat: number; lng: number },
  end: { lat: number; lng: number },
  algorithm: "dijkstra" | "astar",
  trafficLevel: "low" | "medium" | "high",
): Array<[number, number]> {
  const path: Array<[number, number]> = []

  // Add start point
  path.push([start.lat, start.lng])

  // Number of intermediate points depends on the algorithm and traffic
  const pointCount = algorithm === "dijkstra" ? 8 : 6
  const randomFactor = trafficLevel === "low" ? 0.001 : trafficLevel === "medium" ? 0.003 : 0.005

  // Generate intermediate points
  const latStep = (end.lat - start.lat) / (pointCount + 1)
  const lngStep = (end.lng - start.lng) / (pointCount + 1)

  for (let i = 1; i <= pointCount; i++) {
    // Add some randomness to simulate different routes based on algorithm and traffic
    const randomLat = (Math.random() - 0.5) * randomFactor * (algorithm === "dijkstra" ? 1 : 0.7)
    const randomLng = (Math.random() - 0.5) * randomFactor * (algorithm === "dijkstra" ? 1 : 0.7)

    const lat = start.lat + latStep * i + randomLat
    const lng = start.lng + lngStep * i + randomLng

    path.push([lat, lng])
  }

  // Add end point
  path.push([end.lat, end.lng])

  return path
}

// Calculate approximate distance in kilometers
function calculateDistance(path: Array<[number, number]>): number {
  let distance = 0

  for (let i = 0; i < path.length - 1; i++) {
    const [lat1, lng1] = path[i]
    const [lat2, lng2] = path[i + 1]

    // Haversine formula to calculate distance between two points
    const R = 6371 // Earth's radius in km
    const dLat = ((lat2 - lat1) * Math.PI) / 180
    const dLng = ((lng2 - lng1) * Math.PI) / 180
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLng / 2) * Math.sin(dLng / 2)
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    const d = R * c

    distance += d
  }

  return Number.parseFloat(distance.toFixed(2))
}

