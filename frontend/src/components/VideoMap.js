import React, { useEffect, useRef } from 'react';

export default function VideoMap({ gpsPoints, currentTime, onSeek }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null); // google.maps.Map instance
  const markerRef = useRef(null);
  const polylinesRef = useRef([]);
  const clickListenerRef = useRef(null);

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    mapRef.current = new window.google.maps.Map(containerRef.current, {
      center: { lat: 0, lng: 0 },
      zoom: 16,
    });
  }, []);

  // Update polylines / initial marker when gpsPoints change
  useEffect(() => {
    if (!mapRef.current) return;

    // clear old overlays
    polylinesRef.current.forEach(p => p.setMap(null));
    polylinesRef.current = [];
    if (markerRef.current) {
      markerRef.current.setMap(null);
      markerRef.current = null;
    }
    if (clickListenerRef.current) {
      window.google.maps.event.removeListener(clickListenerRef.current);
      clickListenerRef.current = null;
    }

    if (!gpsPoints || gpsPoints.length < 1) return;

    // center map on first point
    const first = gpsPoints[0];
    mapRef.current.setCenter({ lat: first.lat, lng: first.lon });

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
          strokeColor: point.highlight ? 'red' : 'green',
          strokeOpacity: 1.0,
          strokeWeight: 6,
          map: mapRef.current,
        });
      });
    }

    // Create marker at first point
    markerRef.current = new window.google.maps.Marker({
      position: { lat: first.lat, lng: first.lon },
      map: mapRef.current,
      icon: {
        path: window.google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
        scale: 6,
        fillColor: 'blue',
        fillOpacity: 1,
        strokeWeight: 1,
      },
      draggable: false,
    });

    // Add click listener on map to seek video
    clickListenerRef.current = mapRef.current.addListener('click', (e) => {
      const clickedLat = e.latLng.lat();
      const clickedLng = e.latLng.lng();

      let nearestPoint = gpsPoints[0];
      let minDist = Infinity;
      gpsPoints.forEach(point => {
        const dist = Math.hypot(point.lat - clickedLat, point.lon - clickedLng);
        if (dist < minDist) {
          minDist = dist;
          nearestPoint = point;
        }
      });

      if (onSeek) onSeek(nearestPoint.timestamp);
    });

    // cleanup when gpsPoints changes or component unmounts
    return () => {
      polylinesRef.current.forEach(p => p.setMap(null));
      polylinesRef.current = [];
      if (markerRef.current) {
        markerRef.current.setMap(null);
        markerRef.current = null;
      }
      if (clickListenerRef.current) {
        window.google.maps.event.removeListener(clickListenerRef.current);
        clickListenerRef.current = null;
      }
    };
  }, [gpsPoints, onSeek]);

  // Update marker position when currentTime changes (no map re-creation)
  useEffect(() => {
    if (!gpsPoints || gpsPoints.length === 0 || !markerRef.current) return;

    // Find the gps segment containing currentTime
    for (let i = 0; i < gpsPoints.length - 1; i++) {
      const p1 = gpsPoints[i];
      const p2 = gpsPoints[i + 1];
      if (currentTime >= p1.timestamp && currentTime <= p2.timestamp) {
        const ratio = (currentTime - p1.timestamp) / (p2.timestamp - p1.timestamp || 1);
        const lat = p1.lat + (p2.lat - p1.lat) * ratio;
        const lon = p1.lon + (p2.lon - p1.lon) * ratio;
        markerRef.current.setPosition({ lat, lng: lon });
        return;
      }
    }

    // fallback to last point
    const last = gpsPoints[gpsPoints.length - 1];
    markerRef.current.setPosition({ lat: last.lat, lng: last.lon });
  }, [currentTime, gpsPoints]);

  return <div ref={containerRef} style={{ height: '400px', width: '100%' }} />;
}
