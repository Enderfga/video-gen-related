// Copy BibTeX to clipboard
function copyBibtex() {
    const bibtexCode = document.querySelector('.bibtex-box code').textContent;
    navigator.clipboard.writeText(bibtexCode).then(() => {
        const btn = document.querySelector('.copy-btn');
        btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        btn.classList.add('copied');

        setTimeout(() => {
            btn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            btn.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy: ', err);
    });
}

// Lazy load videos when they come into view
document.addEventListener('DOMContentLoaded', function() {
    const videos = document.querySelectorAll('video');

    // Intersection Observer for lazy loading
    const videoObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                video.play().catch(() => {
                    // Autoplay might be blocked
                });
            } else {
                entry.target.pause();
            }
        });
    }, {
        threshold: 0.25
    });

    videos.forEach(video => {
        videoObserver.observe(video);
    });

    // Handle video loading errors gracefully
    videos.forEach(video => {
        video.addEventListener('error', function() {
            const placeholder = document.createElement('div');
            placeholder.style.cssText = `
                width: 100%;
                aspect-ratio: 16/9;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                font-size: 0.9rem;
            `;
            placeholder.innerHTML = '<span>Video placeholder</span>';
            video.parentNode.replaceChild(placeholder, video);
        });
    });

    // Handle image loading errors
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('error', function() {
            this.style.background = 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)';
            this.alt = 'Image placeholder';
        });
    });
});
