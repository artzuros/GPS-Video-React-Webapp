import React, { useEffect, useRef } from 'react';

export default function VideoMap({ gpsPoints, currentTime, onSeek }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const markerRef = useRef(null);
  const polylinesRef = useRef([]);
  const clickListenerRef = useRef(null);
  const hasCenteredRef = useRef(false); // 👈 prevents re-centering every render

  // Initialize map ONCE
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    mapRef.current = new window.google.maps.Map(containerRef.current, {
      center: { lat: 0, lng: 0 },
      zoom: 16,
      mapId: 'DEMO_MAP_ID',
    });
  }, []);

  // Handle gpsPoints updates (draw paths, create marker)
  useEffect(() => {
    if (!mapRef.current || !gpsPoints || gpsPoints.length === 0) return;

    // Clear old polylines
    polylinesRef.current.forEach((p) => p.setMap(null));
    polylinesRef.current = [];

    // Clear old marker
    if (markerRef.current) {
      markerRef.current.map = null;
      markerRef.current = null;
    }

    // Clear old listeners
    if (clickListenerRef.current) {
      window.google.maps.event.removeListener(clickListenerRef.current);
      clickListenerRef.current = null;
    }

    const first = gpsPoints[0];

    // ✅ Only center once (when map first loads)
    if (!hasCenteredRef.current) {
      mapRef.current.setCenter({ lat: first.lat, lng: first.lon });
      hasCenteredRef.current = true;
    }

    // Draw polylines
    if (gpsPoints.length > 1) {
      polylinesRef.current = gpsPoints.slice(0, gpsPoints.length - 1).map((point, i) => {
        const next = gpsPoints[i + 1];
        return new window.google.maps.Polyline({
          path: [
            { lat: point.lat, lng: point.lon },
            { lat: next.lat, lng: next.lon },
          ],
          geodesic: true,
          strokeColor: point.highlight ? 'green' : 'red',
          strokeOpacity: 1.0,
          strokeWeight: 5,
          map: mapRef.current,
        });
      });
    }

    // ✅ Create AdvancedMarkerElement
    const markerDiv = document.createElement('div');
    markerDiv.style.width = '12px';
    markerDiv.style.height = '12px';
    markerDiv.style.background = 'blue';
    markerDiv.style.borderRadius = '50%';
    markerDiv.style.border = '2px solid white';
    markerDiv.style.boxShadow = '0 0 6px rgba(0,0,0,0.4)';

    markerRef.current = new window.google.maps.marker.AdvancedMarkerElement({
      position: { lat: first.lat, lng: first.lon },
      map: mapRef.current,
      content: markerDiv,
    });

    // Add click-to-seek
    clickListenerRef.current = mapRef.current.addListener('click', (e) => {
      const clickedLat = e.latLng.lat();
      const clickedLng = e.latLng.lng();

      let nearestPoint = gpsPoints[0];
      let minDist = Infinity;
      gpsPoints.forEach((point) => {
        const dist = Math.hypot(point.lat - clickedLat, point.lon - clickedLng);
        if (dist < minDist) {
          minDist = dist;
          nearestPoint = point;
        }
      });

      if (onSeek) onSeek(nearestPoint.timestamp);
    });

    return () => {
      polylinesRef.current.forEach((p) => p.setMap(null));
      polylinesRef.current = [];
      if (markerRef.current) {
        markerRef.current.map = null;
        markerRef.current = null;
      }
      if (clickListenerRef.current) {
        window.google.maps.event.removeListener(clickListenerRef.current);
        clickListenerRef.current = null;
      }
    };
  }, [gpsPoints, onSeek]);

  // ✅ Only move marker with time, never reset map
  useEffect(() => {
    if (!gpsPoints || gpsPoints.length === 0 || !markerRef.current) return;

    for (let i = 0; i < gpsPoints.length - 1; i++) {
      const p1 = gpsPoints[i];
      const p2 = gpsPoints[i + 1];
      if (currentTime >= p1.timestamp && currentTime <= p2.timestamp) {
        const ratio = (currentTime - p1.timestamp) / (p2.timestamp - p1.timestamp || 1);
        const lat = p1.lat + (p2.lat - p1.lat) * ratio;
        const lon = p1.lon + (p2.lon - p1.lon) * ratio;
        markerRef.current.position = { lat, lng: lon };
        return;
      }
    }

    // fallback to last point
    const last = gpsPoints[gpsPoints.length - 1];
    markerRef.current.position = { lat: last.lat, lng: last.lon };
  }, [currentTime, gpsPoints]);

  return <div ref={containerRef} style={{ height: '400px', width: '100%' }} />;
}
