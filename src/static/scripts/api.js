/**
 * API utility functions for the labeling system
 */

const API = {
    fetchCurrentFrame: async function () {
        const response = await fetch('/labeling/current_frame');

        if (!response.ok) {
            throw new Error(`Failed to fetch current frame: ${response.status} ${response.statusText}`);
        }

        return response.blob();
    },

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
