// Main JavaScript for MovieLens Recommender System

// On document ready
document.addEventListener('DOMContentLoaded', function() {
    // Activate current navigation link based on URL path
    activateCurrentNavLink();
    
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});

// Activate the appropriate navigation link based on current URL
function activateCurrentNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        
        // Get the href attribute
        const href = link.getAttribute('href');
        
        // Check if the current path matches the link's href
        if (currentPath === href) {
            link.classList.add('active');
        } else if (currentPath.startsWith(href) && href !== '/') {
            // For subpaths, activate parent link
            link.classList.add('active');
        } else if (currentPath === '/' && href === '/') {
            // Home page
            link.classList.add('active');
        }
    });
}

// Function to show a modal alert
function showAlert(message, type = 'info') {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.role = 'alert';
    
    // Add message
    alertDiv.innerHTML = message;
    
    // Add close button
    const closeButton = document.createElement('button');
    closeButton.type = 'button';
    closeButton.className = 'btn-close';
    closeButton.setAttribute('data-bs-dismiss', 'alert');
    closeButton.setAttribute('aria-label', 'Close');
    alertDiv.appendChild(closeButton);
    
    // Find alert container or create one
    let alertContainer = document.getElementById('alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alert-container';
        alertContainer.className = 'container mt-3';
        
        // Insert at the top of the page but below the navbar
        const navbar = document.querySelector('nav');
        if (navbar && navbar.nextSibling) {
            navbar.parentNode.insertBefore(alertContainer, navbar.nextSibling);
        } else {
            document.body.prepend(alertContainer);
        }
    }
    
    // Add the alert to the container
    alertContainer.appendChild(alertDiv);
    
    // Auto-close after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            // Use Bootstrap's dismiss method if available
            if (typeof bootstrap !== 'undefined' && bootstrap.Alert) {
                const bsAlert = new bootstrap.Alert(alertDiv);
                bsAlert.close();
            } else {
                // Manual removal as fallback
                alertDiv.remove();
            }
        }
    }, 5000);
}

// Function to format ratings for display
function formatRating(rating) {
    return parseFloat(rating).toFixed(1);
}

// Function to generate star rating HTML (not used in current design but available for future use)
function generateStarRating(rating) {
    const maxStars = 5;
    const fullStars = Math.floor(rating);
    const halfStar = rating % 1 >= 0.5;
    const emptyStars = maxStars - fullStars - (halfStar ? 1 : 0);
    
    let starsHtml = '';
    
    // Full stars
    for (let i = 0; i < fullStars; i++) {
        starsHtml += '<i class="fas fa-star"></i>';
    }
    
    // Half star
    if (halfStar) {
        starsHtml += '<i class="fas fa-star-half-alt"></i>';
    }
    
    // Empty stars
    for (let i = 0; i < emptyStars; i++) {
        starsHtml += '<i class="far fa-star"></i>';
    }
    
    return starsHtml;
}

// Format numbers for display (add commas for thousands)
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Helper function to truncate text
function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength) + '...';
}

// Helper function to extract year from movie title
function extractYear(title) {
    const match = title.match(/\((\d{4})\)$/);
    return match ? match[1] : null;
}

// Helper function to parse and categorize genres
function parseGenres(genresString) {
    if (!genresString) return [];
    return genresString.split('|').map(genre => genre.trim());
}

// Helper functions for animation
function animateOnScroll() {
    const elements = document.querySelectorAll('.animate-on-scroll');
    
    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const windowHeight = window.innerHeight;
        
        if (elementPosition < windowHeight - 50) {
            element.classList.add('animated');
        }
    });
}

// Add scroll event listener for animations
window.addEventListener('scroll', animateOnScroll);
