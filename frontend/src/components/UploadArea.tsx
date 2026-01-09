import { useState, useCallback, DragEvent } from 'react';
import { Upload, Loader2 } from 'lucide-react';

interface UploadAreaProps {
  onUpload: (files: File[]) => Promise<void>;
  isUploading: boolean;
}

const UploadArea = ({ onUpload, isUploading }: UploadAreaProps) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);

      const files = Array.from(e.dataTransfer.files).filter(
        (file) => {
          // Accept PDFs and images
          const isPDF = file.type === 'application/pdf';
          const isImage = file.type.startsWith('image/');
          // Also check by extension for cases where file.type might be empty
          const fileExtension = file.name.toLowerCase().split('.').pop() || '';
          const imageExtensions = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'];
          const isImageByExtension = imageExtensions.includes(fileExtension);
          
          return isPDF || isImage || isImageByExtension;
        }
      );

      if (files.length > 0) {
        await onUpload(files);
      }
    },
    [onUpload]
  );

  const handleFileSelect = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        await onUpload(files);
      }
      e.target.value = '';
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer ${
        isDragging
          ? 'border-primary bg-primary/10'
          : 'border-sidebar-border hover:border-muted-foreground'
      }`}
    >
      <input
        type="file"
        accept=".pdf,image/*,.png,.jpg,.jpeg,.gif,.webp,.bmp,.svg"
        multiple
        onChange={handleFileSelect}
        disabled={isUploading}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
        id="sidebar-file-upload"
      />
      <label
        htmlFor="sidebar-file-upload"
        className="flex flex-col items-center gap-2 cursor-pointer"
      >
        {isUploading ? (
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        ) : (
          <Upload className="w-8 h-8 text-muted-foreground" />
        )}
        <p className="text-sm font-medium text-foreground">
          {isUploading ? 'Uploading...' : 'Click or drag files here'}
        </p>
        <p className="text-xs text-muted-foreground">Supports PDF documents and images</p>
      </label>
    </div>
  );
};

export default UploadArea;
