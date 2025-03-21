"use client"

import { useRef } from "react"
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMapEvents } from "react-leaflet"
import "leaflet/dist/leaflet.css"
import "leaflet-defaulticon-compatibility"
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css"
import L from "leaflet"

interface MapProps {
  startPoint: { lat: number; lng: number }
  endPoint: { lat: number; lng: number }
  path: Array<[number, number]>
  setStartPoint: (point: { lat: number; lng: number }) => void
  setEndPoint: (point: { lat: number; lng: number }) => void
}

// Custom marker icons
const createCustomIcon = (color: string) => {
  return L.divIcon({
    className: "custom-icon",
    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 4px rgba(0,0,0,0.4);"></div>`,
    iconSize: [24, 24],
    iconAnchor: [12, 12],
  })
}

const startIcon = createCustomIcon("#22c55e") // Green
const endIcon = createCustomIcon("#ef4444") // Red

// Component to handle map clicks
function MapClickHandler({
  setStartPoint,
  setEndPoint,
}: {
  setStartPoint: (point: { lat: number; lng: number }) => void
  setEndPoint: (point: { lat: number; lng: number }) => void
}) {
  const isSettingStart = useRef(true)

  const map = useMapEvents({
    click: (e) => {
      const { lat, lng } = e.latlng

      if (isSettingStart.current) {
        setStartPoint({ lat, lng })
        isSettingStart.current = false
      } else {
        setEndPoint({ lat, lng })
        isSettingStart.current = true
      }
    },
  })

  return null
}

export default function Map({ startPoint, endPoint, path, setStartPoint, setEndPoint }: MapProps) {
  // Center the map on the midpoint between start and end
  const center: [number, number] = [(startPoint.lat + endPoint.lat) / 2, (startPoint.lng + endPoint.lng) / 2]

  // Calculate appropriate zoom level based on distance
  const getZoomLevel = () => {
    const latDiff = Math.abs(startPoint.lat - endPoint.lat)
    const lngDiff = Math.abs(startPoint.lng - endPoint.lng)
    const maxDiff = Math.max(latDiff, lngDiff)

    if (maxDiff > 1) return 9
    if (maxDiff > 0.5) return 10
    if (maxDiff > 0.1) return 12
    if (maxDiff > 0.05) return 13
    return 14
  }

  return (
    <MapContainer
      center={center}
      zoom={getZoomLevel()}
      style={{ height: "600px", width: "100%", borderRadius: "0.375rem" }}
      className="z-0"
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <MapClickHandler setStartPoint={setStartPoint} setEndPoint={setEndPoint} />

      <Marker position={[startPoint.lat, startPoint.lng]} icon={startIcon}>
        <Popup>
          Start Point
          <br />
          {startPoint.lat.toFixed(4)}, {startPoint.lng.toFixed(4)}
        </Popup>
      </Marker>

      <Marker position={[endPoint.lat, endPoint.lng]} icon={endIcon}>
        <Popup>
          End Point
          <br />
          {endPoint.lat.toFixed(4)}, {endPoint.lng.toFixed(4)}
        </Popup>
      </Marker>

      {path.length > 0 && <Polyline positions={path} color="#3b82f6" weight={5} opacity={0.7} dashArray={[10, 5]} />}
    </MapContainer>
  )
}

