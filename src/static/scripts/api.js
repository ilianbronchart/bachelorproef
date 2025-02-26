/**
 * API utility functions for the labeling system
 */

const API = {
    /**
     * Seek to a specific frame in the recording
     * If no frameIndex is provided, returns the current frame
     * @param {number} [frameIndex] - The frame index to seek to
     * @returns {Promise<Blob>} - Promise resolving to the frame image blob
     */
    seekFrame: async function (frameIndex) {
        const url = frameIndex !== undefined && frameIndex !== null ?
            `/labeling/seek?frame_idx=${frameIndex}` :
            '/labeling/seek';

        const response = await fetch(url);

        if (!response.ok) {
            const frameDesc = frameIndex !== undefined ? frameIndex : 'current';
            throw new Error(`Failed to seek to frame ${frameDesc}: ${response.status} ${response.statusText}`);
        }

        return response.blob();
    },

    /**
     * Get point labels for a specific frame
     * @param {number} frameIndex - The frame index to get labels for
     * @returns {Promise<Object>} - Promise resolving to the point labels data
     */
    getPointLabels: async function () {
        const response = await fetch(`/labeling/point_labels`);
        if (!response.ok) {
            throw new Error(`Failed to fetch point labels for frame ${frameIndex}: ${response.status} ${response.statusText}`);
        }
        return response.json();
    },

    /**
     * Post a point annotation to the server
     * @param {Object} annotationData - The annotation data to post
     * @param {Array} annotationData.point - [x, y] coordinates of the point
     * @param {number} annotationData.label - 1 for positive, 0 for negative label
     * @param {number} annotationData.frameIdx - The frame index for the annotation
     * @param {number} annotationData.classId - The class ID for the annotation
     * @param {boolean} annotationData.deletePoint - Whether to delete the point instead of adding
     * @returns {Promise<Blob>} - Promise resolving to the updated frame image blob
     */
    postAnnotation: async function (annotationData) {
        const response = await fetch('/labeling/annotations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                point: annotationData.point,
                label: annotationData.label,
                frame_idx: annotationData.frameIdx,
                class_id: annotationData.classId,
                delete_point: annotationData.deletePoint
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to post annotation: ${response.status} ${response.statusText}`);
        }

        return response.blob();
    }
};

// Make API available globally
window.API = API;
