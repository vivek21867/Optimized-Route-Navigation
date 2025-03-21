"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Code } from "lucide-react"

interface AlgorithmExplanationProps {
  algorithm: "dijkstra" | "astar"
}

export default function AlgorithmExplanation({ algorithm }: AlgorithmExplanationProps) {
  const [showCode, setShowCode] = useState(false)

  return (
    <div className="space-y-4">
      {algorithm === "dijkstra" ? (
        <>
          <p className="text-sm text-gray-600">
            Dijkstra's algorithm finds the shortest path between nodes in a graph by:
          </p>
          <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
            <li>Assigning initial distance values</li>
            <li>Visiting the unvisited vertex with the smallest distance</li>
            <li>Updating distances to adjacent vertices</li>
            <li>Repeating until the destination is reached</li>
          </ol>

          <div className="mt-4">
            <Button variant="outline" size="sm" onClick={() => setShowCode(!showCode)} className="w-full text-xs">
              <Code className="mr-2 h-3 w-3" />
              {showCode ? "Hide Code" : "Show Implementation"}
            </Button>

            {showCode && (
              <pre className="mt-2 p-3 bg-gray-100 rounded-md text-xs overflow-x-auto">
                <code className="text-gray-800">
                  {`function dijkstra(graph, startNode, endNode) {
  // Set distances for all nodes to Infinity except start node
  const distances = {};
  const previous = {};
  const nodes = new Set();
  
  for (let vertex in graph) {
    distances[vertex] = Infinity;
    previous[vertex] = null;
    nodes.add(vertex);
  }
  distances[startNode] = 0;
  
  while (nodes.size > 0) {
    // Find node with minimum distance
    let minNode = null;
    for (let node of nodes) {
      if (minNode === null || distances[node] < distances[minNode]) {
        minNode = node;
      }
    }
    
    // If we reached the end node or no path exists
    if (minNode === endNode || distances[minNode] === Infinity) {
      break;
    }
    
    nodes.delete(minNode);
    
    // Update distances to neighbors
    for (let neighbor in graph[minNode]) {
      const alt = distances[minNode] + graph[minNode][neighbor];
      if (alt < distances[neighbor]) {
        distances[neighbor] = alt;
        previous[neighbor] = minNode;
      }
    }
  }
  
  // Reconstruct path
  const path = [];
  let current = endNode;
  
  while (current !== null) {
    path.unshift(current);
    current = previous[current];
  }
  
  return path;
}`}
                </code>
              </pre>
            )}
          </div>
        </>
      ) : (
        <>
          <p className="text-sm text-gray-600">
            A* (A-Star) algorithm improves on Dijkstra by using heuristics to guide the search:
          </p>
          <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
            <li>Uses a priority queue based on f(n) = g(n) + h(n)</li>
            <li>g(n) is the cost from start to current node</li>
            <li>h(n) is the estimated cost from current to goal</li>
            <li>Prioritizes paths that seem to lead toward the goal</li>
          </ol>

          <div className="mt-4">
            <Button variant="outline" size="sm" onClick={() => setShowCode(!showCode)} className="w-full text-xs">
              <Code className="mr-2 h-3 w-3" />
              {showCode ? "Hide Code" : "Show Implementation"}
            </Button>

            {showCode && (
              <pre className="mt-2 p-3 bg-gray-100 rounded-md text-xs overflow-x-auto">
                <code className="text-gray-800">
                  {`function aStar(graph, startNode, endNode, heuristic) {
  // Priority queue for nodes to visit
  const openSet = new PriorityQueue();
  openSet.enqueue(startNode, 0);
  
  // Track where we came from
  const cameFrom = {};
  
  // Cost from start to node
  const gScore = {};
  for (let node in graph) {
    gScore[node] = Infinity;
  }
  gScore[startNode] = 0;
  
  // Estimated total cost from start to goal through node
  const fScore = {};
  for (let node in graph) {
    fScore[node] = Infinity;
  }
  fScore[startNode] = heuristic(startNode, endNode);
  
  while (!openSet.isEmpty()) {
    const current = openSet.dequeue().element;
    
    if (current === endNode) {
      return reconstructPath(cameFrom, current);
    }
    
    for (let neighbor in graph[current]) {
      // Distance from start to neighbor through current
      const tentativeGScore = gScore[current] + graph[current][neighbor];
      
      if (tentativeGScore < gScore[neighbor]) {
        // This path is better than any previous one
        cameFrom[neighbor] = current;
        gScore[neighbor] = tentativeGScore;
        fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, endNode);
        
        if (!openSet.contains(neighbor)) {
          openSet.enqueue(neighbor, fScore[neighbor]);
        }
      }
    }
  }
  
  // No path found
  return [];
}

function reconstructPath(cameFrom, current) {
  const path = [current];
  while (cameFrom[current]) {
    current = cameFrom[current];
    path.unshift(current);
  }
  return path;
}`}
                </code>
              </pre>
            )}
          </div>
        </>
      )}

      <div className="mt-4 p-3 bg-blue-50 rounded-md border border-blue-100">
        <h4 className="text-sm font-medium text-blue-800 mb-1">Traffic Integration</h4>
        <p className="text-xs text-gray-600">
          In our implementation, traffic conditions modify edge weights in the graph. Higher traffic increases the
          "cost" of traveling certain routes, causing the algorithm to potentially select longer but less congested
          paths.
        </p>
      </div>
    </div>
  )
}

