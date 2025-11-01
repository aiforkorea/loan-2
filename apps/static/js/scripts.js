// apps/static/js/scripts.js 
// API 키 복사 기능
document.addEventListener('DOMContentLoaded', function() {
    // API 키 복사 버튼 기능
    const copyButtons = document.querySelectorAll('.copy-api-key');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const apiKeyString = this.getAttribute('data-api-key');
            navigator.clipboard.writeText(apiKeyString).then(() => {
                alert('API 키가 클립보드에 복사되었습니다!');
            }).catch(err => {
                console.error('API 키 복사 실패: ', err);
            });
        });
    });
});
